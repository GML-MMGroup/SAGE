"""
脚本作用：
共享 VLM 物体框定原语——给定「属性/方位描述 + 图片」，调用多个 VLM(gpt-5.5 / qwen-vl-max / qwen-vl-plus)
返回该物体的像素 bbox，并做 IoU 投票得共识框；另含 vlm_judge(多候选图选最佳)。供各定位方案复用。
跑在 egolink base python；VAPI/QWEN key、base URL 与代理均通过环境变量配置。

执行逻辑：
1. vlm_boxes(desc, frame)：每个模型回一个 bbox(像素，看不到=None)，解析其 JSON。
2. vote_box(boxes)：按 IoU≥0.3 聚类，取最大簇均值作共识框；无共识(各说各话)→返回 None+低置信标记。
3. vlm_judge(desc, crops)：每个模型从候选裁剪图里选最像描述的一张，多数票定胜出 index。
"""
import os
import re
import base64

MODELS = ["gpt-5.5", "qwen-vl-max", "qwen-vl-plus"]
_QWEN = {"qwen-vl-max", "qwen-vl-plus", "qwen-vl-max-latest"}


def _b64(path):
    """图片转 base64，输入：路径 -> 输出：base64 串"""
    return base64.b64encode(open(path, "rb").read()).decode()


def _parse_bbox(s):
    """从回复抽 bbox(像素)，输入：文本 -> 输出：[x1,y1,x2,y2] 或 None"""
    m = re.search(r'"?bbox"?\s*[:=]\s*\[([^\]]+)\]', str(s))
    if not m:
        m = re.search(r'\[\s*(\d[\d.,\s]+\d)\s*\]', str(s))
        if not m:
            return None
        nums = [float(x) for x in re.findall(r"[-\d.]+", m.group(1))]
    else:
        nums = [float(x) for x in re.findall(r"[-\d.]+", m.group(1))]
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _vlm_call(model, content):
    """统一调用 VLM，输入：模型名/OpenAI content -> 输出：回复文本"""
    msgs = [{"role": "user", "content": content}]
    if model in _QWEN:
        from openai import OpenAI
        cli = OpenAI(api_key=os.environ["QWEN_API_KEY"], base_url=os.environ["QWEN_API_URL"])
        r = cli.chat.completions.create(model=model, messages=msgs, temperature=0)
        input_tokens = 0
        output_tokens = 0
        if hasattr(r, "usage") and r.usage:
            input_tokens = getattr(r.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(r.usage, "completion_tokens", 0) or 0
        _add_frame_boxer_usage(input_tokens, output_tokens)
        return str(r.choices[0].message.content)
    # 默认走 egolink 的 gpt-5.5(VAPI)
    from config.visual_agent_config import call_visual_agent_model
    r, input_tokens, output_tokens = call_visual_agent_model(msgs)
    _add_frame_boxer_usage(input_tokens, output_tokens)
    return str(r)


def _add_frame_boxer_usage(input_tokens, output_tokens):
    """Accumulate VLM usage into the active retail1 frame boxer stage if profiling is enabled."""
    try:
        from run import stage_latency
        stage_latency.add_usage("frame_boxer", input_tokens, output_tokens)
    except Exception:
        return


def ask_box(model, desc, frame_path, W, H):
    """单模型框定，输入：模型/描述/帧/宽高 -> 输出：(bbox 或 None, 原始文本)"""
    prompt = (f"这是一帧照片，分辨率 {W}x{H} 像素(左上=0,0)。请框出：{desc}。"
              "注意框住物体本体(实物)，不要框价签/文字。"
              f'只输出 JSON：{{"bbox":[x1,y1,x2,y2]}}，整数像素坐标，x∈[0,{W}],y∈[0,{H}]；看不到填 null。')
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(frame_path)}"}}]
    try:
        raw = _vlm_call(model, content)
        return _parse_bbox(raw), raw
    except Exception as e:
        return None, f"ERROR {str(e)[:120]}"


def _iou(a, b):
    """两框 IoU，输入：两个 bbox -> 输出：交并比"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def vote_box(model_boxes, iou_thr=0.3):
    """IoU 投票得共识框，输入：[(model,bbox)]/IoU阈 -> 输出：(共识bbox 或 None, 置信dict)。
    最大互相 IoU≥阈的簇 → 取簇内均值；簇大小≥2 视为有共识。"""
    items = [(m, b) for m, b in model_boxes if b]
    if not items:
        return None, {"agree": 0, "n": 0, "members": []}
    if len(items) == 1:
        return items[0][1], {"agree": 1, "n": 1, "members": [items[0][0]]}
    best_cluster = []
    for i, (mi, bi) in enumerate(items):
        cluster = [(mi, bi)]
        for j, (mj, bj) in enumerate(items):
            if j != i and _iou(bi, bj) >= iou_thr:
                cluster.append((mj, bj))
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    boxes = [b for _, b in best_cluster]
    mean = [sum(b[k] for b in boxes) / len(boxes) for k in range(4)]
    return mean, {"agree": len(best_cluster), "n": len(items),
                  "members": [m for m, _ in best_cluster]}


def vlm_boxes(desc, frame_path, W, H, models=None):
    """多模型框定，输入：描述/帧/宽高/模型表 -> 输出：[(model,bbox,raw)]"""
    out = []
    for m in (models or MODELS):
        box, raw = ask_box(m, desc, frame_path, W, H)
        out.append((m, box, raw))
    return out


def vlm_judge(desc, crop_paths, models=None):
    """多候选裁剪图选最佳，输入：描述/裁剪图路径表/模型表 -> 输出：(胜出index, 每模型投票dict)。
    每个模型回 {"best": i}；多数票；平票取第一个被投的。"""
    n = len(crop_paths)
    prompt = (f"下面给出 {n} 张候选物体裁剪图(按 0..{n-1} 顺序)。哪一张最符合描述「{desc}」"
              "且是物体本体(不是价签/文字)？只输出 JSON：{\"best\": 序号整数}。")
    content = [{"type": "text", "text": prompt}]
    for i, p in enumerate(crop_paths):
        content.append({"type": "text", "text": f"[{i}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(p)}"}})
    votes = {}
    for m in (models or MODELS):
        try:
            raw = _vlm_call(m, content)
            mm = re.search(r'"?best"?\s*[:=]\s*(\d+)', str(raw))
            if mm:
                votes[m] = int(mm.group(1)) % n
        except Exception:
            pass
    if not votes:
        return 0, votes
    from collections import Counter
    best = Counter(votes.values()).most_common(1)[0][0]
    return best, votes
