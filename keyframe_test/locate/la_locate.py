#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""la_locate.py - LocateAnything-3B referring-grounding batch CLI.

Runs UNDER the isolated LA venv (transformers==4.57.1):
    $LA_PYTHON keyframe_test/locate/la_locate.py \
        --jobs jobs.json --out boxes.json

input  : jobs.json = [{"id","image","query"}, ...]  (query may be Chinese or English)
output : boxes.json = [{"id","image","query","box":[x1,y1,x2,y2]|null,"score","raw"}, ...]
          box coords are PIXELS in the source image; null when the model says no-object.
purpose: load LocateAnything-3B ONCE and ground every job (referring, single instance),
         atomic + resumable so a crash never loses finished work.
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

# --- locate the official worker + the model -------------------------------------------------
# Local defaults (override via env). The official Eagle worker is optional; the local
# LocateAnything release also supports the AutoModel fallback below.
_RL_ROOT = Path(__file__).resolve().parents[4]
EAGLE_EMBODIED = os.environ.get("LA_EAGLE_EMBODIED", str(_RL_ROOT / "Eagle" / "Embodied"))
MODEL_PATH = os.environ.get("LA_MODEL_PATH", str(_RL_ROOT / "models" / "LocateAnything-3B"))

# Referring-grounding prompt template (single instance). Per the model card / worker.
PROMPT_TEMPLATE = "Locate a single instance that matches the following description: {phrase}."

_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_first_box(answer, w, h):
    """model answer + image (w,h) -> first [x1,y1,x2,y2] in pixels, or None.

    LocateAnything emits <box><x1><y1><x2><y2></box> with coords in [0,1000]; it emits
    <box>none</box> when nothing matches. We take the first box (single-instance grounding).
    """
    m = _BOX_RE.search(answer or "")
    if not m:
        return None
    x1, y1, x2, y2 = (int(g) for g in m.groups())
    box = [x1 / 1000.0 * w, y1 / 1000.0 * h, x2 / 1000.0 * w, y2 / 1000.0 * h]
    # guard against degenerate / inverted boxes
    bx1, by1, bx2, by2 = box
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1
    if (bx2 - bx1) < 1 or (by2 - by1) < 1:
        return None
    return [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)]


def load_worker():
    """env -> a loaded LocateAnythingWorker (preferred) or a tiny AutoModel fallback shim.

    Returns an object exposing .ground(image, phrase, **kw) -> raw_answer_str.
    """
    import torch
    from PIL import Image  # noqa: F401  (imported for side-effect parity / sanity)

    # Prefer the official worker shipped in Eagle/Embodied.
    if os.path.isdir(EAGLE_EMBODIED) and EAGLE_EMBODIED not in sys.path:
        sys.path.insert(0, EAGLE_EMBODIED)
    try:
        from locateanything_worker import LocateAnythingWorker

        w = LocateAnythingWorker(MODEL_PATH, device="cuda", dtype=torch.bfloat16)

        class _OfficialShim:
            backend = "official_worker"

            def ground(self, image, phrase, **kw):
                res = w.ground_single(image, phrase, verbose=False, **kw)
                return res["answer"] if isinstance(res, dict) else str(res)

        return _OfficialShim()
    except Exception:
        sys.stderr.write("[la_locate] official worker import failed, using AutoModel fallback:\n")
        traceback.print_exc()

    # --- AutoModel fallback (HF model-card path) -------------------------------------------
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .to("cuda")
        .eval()
    )

    class _AutoShim:
        backend = "automodel_fallback"

        @torch.no_grad()
        def ground(self, image, phrase, max_new_tokens=2048, **kw):
            prompt = PROMPT_TEMPLATE.format(phrase=phrase)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
            pixel_values = inputs["pixel_values"].to(torch.bfloat16)
            resp = model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.1,
                verbose=False,
            )
            return resp[0] if isinstance(resp, tuple) else resp

    return _AutoShim()


def _atomic_write(path, obj):
    """obj + path -> json written atomically (write .tmp then os.replace)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="LocateAnything-3B referring grounding (batch).")
    ap.add_argument("--jobs", required=True, help="jobs.json = [{id,image,query}]")
    ap.add_argument("--out", required=True, help="boxes.json output path")
    ap.add_argument("--prompt-lang", choices=["raw", "en"], default="raw",
                    help="raw=use query as-is (works for Chinese); en=expects pre-translated English")
    ap.add_argument("--retries", type=int, default=3, help="per-job retries before skipping")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list):
        raise SystemExit("jobs file must be a JSON list of {id,image,query}")

    # Resume: keep already-done ids (those with a non-error 'raw' field present).
    done = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                for row in json.load(f):
                    if row.get("id") is not None and "raw" in row and not str(row.get("raw", "")).startswith("Error:"):
                        done[row["id"]] = row
        except Exception:
            done = {}

    todo = [j for j in jobs if j.get("id") not in done]
    sys.stderr.write(f"[la_locate] {len(jobs)} jobs, {len(done)} already done, {len(todo)} to run\n")
    sys.stderr.flush()

    from PIL import Image

    worker = None
    if todo:
        t0 = time.time()
        worker = load_worker()
        sys.stderr.write(f"[la_locate] backend={worker.backend} model load {time.time()-t0:.1f}s\n")
        sys.stderr.flush()

    results = dict(done)
    latencies = []
    for i, job in enumerate(todo):
        jid = job.get("id")
        img_path = job.get("image")
        query = job.get("query", "")
        row = {"id": jid, "image": img_path, "query": query, "box": None, "score": None, "raw": ""}
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            ok = False
            for attempt in range(args.retries):
                try:
                    ts = time.time()
                    answer = worker.ground(image, query, max_new_tokens=args.max_new_tokens)
                    dt = time.time() - ts
                    latencies.append(dt)
                    row["raw"] = answer
                    row["box"] = parse_first_box(answer, w, h)
                    ok = True
                    sys.stderr.write(f"[la_locate] {i+1}/{len(todo)} id={jid} {dt:.2f}s box={row['box']}\n")
                    sys.stderr.flush()
                    break
                except Exception as e:  # retry transient failures
                    sys.stderr.write(f"[la_locate] id={jid} attempt {attempt+1} failed: {e}\n")
                    sys.stderr.flush()
                    time.sleep(2 ** attempt)
            if not ok:
                row["raw"] = "Error: all retries failed"
        except Exception as e:
            row["raw"] = f"Error: {e}"
            sys.stderr.write(f"[la_locate] id={jid} FATAL: {e}\n")
        results[jid] = row
        # incremental atomic save so a crash keeps finished work
        _atomic_write(args.out, [results.get(j.get("id"), {"id": j.get("id")}) for j in jobs])

    # final ordered write
    _atomic_write(args.out, [results.get(j.get("id"), {"id": j.get("id")}) for j in jobs])
    if latencies:
        avg = sum(latencies) / len(latencies)
        sys.stderr.write(f"[la_locate] done. avg latency {avg:.2f}s over {len(latencies)} queries\n")
    sys.stderr.write(f"[la_locate] wrote {args.out}\n")


if __name__ == "__main__":
    main()
