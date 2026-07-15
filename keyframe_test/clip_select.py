"""
脚本作用：
封装 Chinese-CLIP 的加载与图文编码：批量编码候选帧与中文目标短语（各自 L2 归一化），
用余弦相似度为每个中文目标选出最相关的 top-k 帧。供 run_test.py 每个视频只编码一次后复用。

执行逻辑：
1. 加载 Chinese-CLIP（优先本地目录 CLIP_MODEL，回退 HF 仓库名）到 GPU/CPU。
2. encode_images：分批读图→get_image_features→归一化，返回 [N,D] 图特征。
3. encode_texts：中文短语→get_text_features→归一化；topk：文本@图特征.T 取最高分帧。

运行示例：
    CLIP_MODEL=processed/models/chinese-clip-vit-base-patch16 \
      python keyframe_test/clip_select.py   # 加载自测：打印模型与特征维度
"""
import os
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get(
    "CLIP_MODEL",
    str(PROJECT_ROOT / "processed" / "models" / "chinese-clip-vit-base-patch16"),
)

class ClipSelector:
    """Chinese-CLIP 关键帧选择器：编码帧/中文文本并按余弦相似度取 top-k。
    """

    def __init__(self, model_dir=DEFAULT_MODEL, device=None, batch_size=64):
        """初始化，输入：模型目录/设备/批大小 -> 输出：无。"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        src = model_dir if Path(model_dir).exists() else "OFA-Sys/chinese-clip-vit-base-patch16"
        self.model_src = src
        self.model = ChineseCLIPModel.from_pretrained(src).to(self.device).eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(src)
        if self.device == "cuda":
            self.model = self.model.half()  # fp16 推理提速

    @staticmethod
    def _as_embed(out):
        """取投影后的 CLIP 嵌入，输入：get_*_features 返回值 -> 输出：嵌入张量（兼容 transformers 5.x 输出对象）"""
        return out.pooler_output if hasattr(out, "pooler_output") else out

    @torch.no_grad()
    def encode_images(self, image_paths):
        """批量编码候选帧，输入：帧路径列表或 PIL.Image 列表 -> 输出：归一化图特征张量 [N,D]"""
        feats = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start:start + self.batch_size]
            images = [(p if isinstance(p, Image.Image) else Image.open(p)).convert("RGB")
                      for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
            batch_feat = self._as_embed(self.model.get_image_features(**inputs))
            batch_feat = batch_feat / batch_feat.norm(dim=-1, keepdim=True)
            feats.append(batch_feat.float())
        return torch.cat(feats, dim=0) if feats else torch.empty(0)

    @torch.no_grad()
    def encode_texts(self, texts):
        """编码中文目标短语，输入：文本列表 -> 输出：归一化文本特征张量 [M,D]"""
        inputs = self.processor(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
        feat = self._as_embed(self.model.get_text_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float()

    @torch.no_grad()
    def topk(self, text, image_feats, k=5):
        """为单个中文目标取 top-k 帧，输入：目标文本/图特征/k -> 输出：[(帧下标, 相似度)]（降序）"""
        txt_feat = self.encode_texts([text])           # [1,D]
        sims = (txt_feat @ image_feats.T)[0]           # [N]
        k = min(k, sims.shape[0])
        scores, idx = torch.topk(sims, k)
        return [(int(i), float(s)) for i, s in zip(idx.tolist(), scores.tolist())]


if __name__ == "__main__":
    sel = ClipSelector()
    print(f"已加载模型：{sel.model_src}  device={sel.device}")
    feat = sel.encode_texts(["亮蓝色盘炸物柠檬角", "双手包饺子"])
    print(f"文本特征维度：{tuple(feat.shape)}")
