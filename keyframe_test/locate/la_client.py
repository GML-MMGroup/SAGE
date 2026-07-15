#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""la_client.py - BASE-env client for LocateAnything-3B grounding.

The LA model lives in an isolated venv (transformers==4.57.1) that is incompatible with
the base env's Chinese-CLIP (transformers 5.x). This client lets base-env code ground
objects WITHOUT importing the model: it writes a temp jobs.json, runs `la_locate.py` under
the LA venv as a subprocess, and reads boxes.json back.

    from keyframe_test.locate.la_client import locate, locate_batch
    box = locate("/abs/frame.jpg", "orange heart-shaped cookie box")  # -> [x1,y1,x2,y2] | None
    rows = locate_batch([{"id":"a","image":"/abs/x.jpg","query":"海鲜"}])  # -> list of result rows
"""
import json
import os
import subprocess
import sys
import tempfile
import uuid
import urllib.request
import urllib.error

# Local infra paths. Override via env if needed.
_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
LA_PYTHON = os.environ.get("LA_PYTHON", sys.executable)
LA_LOCATE = os.environ.get(
    "LA_LOCATE_SCRIPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "la_locate.py"),
)
LA_MODEL_PATH = os.environ.get("LA_MODEL_PATH", os.path.join(_RL_ROOT, "models", "LocateAnything-3B"))
LA_EAGLE_EMBODIED = os.environ.get("LA_EAGLE_EMBODIED", os.path.join(_RL_ROOT, "Eagle", "Embodied"))

# Persistent service (loads the 22GB model once). When healthy, locate_batch POSTs to it;
# otherwise it transparently falls back to the per-call subprocess below. Set LA_SERVICE_URL=""
# to force the subprocess path.
LA_SERVICE_URL = os.environ.get("LA_SERVICE_URL", "http://127.0.0.1:8731")
# Per-request HTTP timeout to the service (generous: one grounding can take tens of seconds and
# may queue behind other requests). Clients block here, FIFO-serialized server-side.
LA_HTTP_TIMEOUT = float(os.environ.get("LA_HTTP_TIMEOUT", "600"))
LA_HTTP_RETRIES = int(os.environ.get("LA_HTTP_RETRIES", "2"))

# A urllib opener that bypasses any http(s)_proxy/ALL_PROXY env for localhost — these are
# loopback calls and must never go through the international proxy.
_NOPROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _service_healthy(base_url, timeout=3.0):
    """ping GET /health -> True if the LA service answers 200 with status ok, else False.
    输入：服务基址/超时 -> 输出：bool。任何异常都视为不健康（触发子进程回退）。"""
    if not base_url:
        return False
    try:
        with _NOPROXY_OPENER.open(base_url.rstrip("/") + "/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("status") == "ok"
    except Exception:
        return False


def _locate_via_service(base_url, norm, prompt_lang):
    """POST each normalized job to the service /locate (FIFO server-side) -> result rows.
    输入：服务基址/规范化jobs/prompt_lang -> 输出：[{id,image,query,box,score,raw}]。
    单条按 HTTP 重试 LA_HTTP_RETRIES 次；最终失败的行带 'Error:' raw（不抛、不丢）。"""
    url = base_url.rstrip("/") + "/locate"
    rows = []
    for j in norm:
        payload = json.dumps({"id": j["id"], "image": j["image"], "query": j["query"],
                              "prompt_lang": prompt_lang}).encode("utf-8")
        result = None
        for attempt in range(LA_HTTP_RETRIES + 1):
            try:
                req = urllib.request.Request(url, data=payload,
                                             headers={"Content-Type": "application/json"})
                with _NOPROXY_OPENER.open(req, timeout=LA_HTTP_TIMEOUT) as r:
                    result = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:
                if attempt >= LA_HTTP_RETRIES:
                    result = {"id": j["id"], "box": None, "score": None,
                              "raw": f"Error: service request failed: {e}"}
        rows.append({"id": j["id"], "image": j["image"], "query": j["query"],
                     "box": result.get("box"), "score": result.get("score"),
                     "raw": result.get("raw", "")})
    return rows


def locate_batch(jobs, prompt_lang="raw", timeout=3600, workdir=None):
    """jobs (list of {id,image,query}) -> list of result rows [{id,image,query,box,score,raw}].

    Prefers the persistent LA service (POST /locate, model loaded once). If the service is
    unreachable/unhealthy it transparently spawns la_locate.py under the LA venv (the original
    per-call subprocess path). box is pixel [x1,y1,x2,y2] or None. Never raises on a single
    grounding failure; rows carry an "Error:" raw string instead. Raises only if the subprocess
    infrastructure itself is broken (bad venv / script path) AND no service is available.
    """
    if not jobs:
        return []
    # normalize / assign ids
    norm = []
    for i, j in enumerate(jobs):
        norm.append({
            "id": j.get("id", f"j{i}"),
            "image": os.path.abspath(j["image"]),
            "query": j["query"],
        })

    # --- fast path: persistent service (no model reload) -----------------------------------
    if _service_healthy(LA_SERVICE_URL):
        return _locate_via_service(LA_SERVICE_URL, norm, prompt_lang)

    # --- fallback: original per-call subprocess --------------------------------------------
    wd = workdir or tempfile.mkdtemp(prefix="la_client_")
    os.makedirs(wd, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    jobs_path = os.path.join(wd, f"jobs_{tag}.json")
    out_path = os.path.join(wd, f"boxes_{tag}.json")
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(norm, f, ensure_ascii=False)

    env = dict(os.environ)
    env["LA_MODEL_PATH"] = LA_MODEL_PATH
    env["LA_EAGLE_EMBODIED"] = LA_EAGLE_EMBODIED
    env.setdefault("HF_HOME", os.path.join(_RL_ROOT, "models", ".hf_home"))
    # offline: weights are local, avoid any hub call slowing things down
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    cmd = [LA_PYTHON, LA_LOCATE, "--jobs", jobs_path, "--out", out_path,
           "--prompt-lang", prompt_lang]
    proc = subprocess.run(cmd, env=env, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not os.path.exists(out_path):
        raise RuntimeError(
            f"la_locate subprocess produced no output (rc={proc.returncode}).\n"
            f"STDERR tail:\n{proc.stderr[-2000:]}"
        )
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def locate(image_path, query, prompt_lang="raw", timeout=600):
    """single (image_path, query) -> [x1,y1,x2,y2] pixel box or None.

    Convenience one-shot wrapper around locate_batch. Returns None on no-object / failure.
    """
    rows = locate_batch([{"id": "single", "image": image_path, "query": query}],
                        prompt_lang=prompt_lang, timeout=timeout)
    if not rows:
        return None
    return rows[0].get("box")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        print(locate(sys.argv[1], sys.argv[2]))
    else:
        print("usage: python la_client.py <image_path> <query>")
