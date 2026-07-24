"""CLIP 语义匹配模块 — Phase 1 第三层漏斗核心。

目标：用 CLIP 视觉语义理解判断视频帧与版权图是否为同一产品。

设计：
  1. 启动时预计算公司图库的所有 CLIP embedding（缓存到 pkl）
  2. 匹配时只需对单帧做一次 embedding
  3. 余弦相似度排序取 Top N
"""
import pickle
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


@dataclass
class CLIPMatch:
    """CLIP 匹配结果。"""
    frame_path: Path
    frame_idx: int
    company_path: str
    company_name: str
    similarity: float
    score: float  # similarity * 100


class CLIPMatcher:
    """CLIP 语义匹配器。

    核心流程：
      1. load_library() → 预计算所有公司图的 CLIP embedding
      2. match_frame() → 对视频帧编码，与图库做余弦相似度
      3. 返回 Top N 候选
    """

    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 threshold: float = 0.25,
                 top_n: int = 5,
                 cache_dir: str = "./fingerprints"):
        """
        Args:
            model_name: CLIP 模型架构
            pretrained: 预训练权重
            threshold: 相似度阈值，低于此值不返回
            top_n: 返回前 N 个候选
            cache_dir: embedding 缓存目录
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.threshold = threshold
        self.top_n = top_n
        self.cache_dir = Path(cache_dir)

        self._model = None
        self._preprocess = None
        self._device = None

        # 图库 {path: embedding}
        self._embeddings: dict[str, np.ndarray] = {}
        self._image_paths: list[str] = []

    def _init_model(self):
        """初始化 CLIP 模型（懒加载）。"""
        if self._model is not None:
            return

        if not CLIP_AVAILABLE:
            raise RuntimeError(
                "open-clip-torch 未安装: pip install open-clip-torch torch torchvision"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self._device,
        )
        self._model.eval()
        print(f"[CLIP] 模型加载完成: {self.model_name} (device={self._device})")

    def load_library(self, full_dir: str, crop_dir: str = None) -> int:
        """加载公司图库，预计算 CLIP embedding。

        优先从缓存加载，缓存不存在时重新计算。

        Returns:
            加载的图片数量
        """
        self._init_model()

        # 收集所有图片路径
        paths = []
        full_path = Path(full_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            paths += list(full_path.glob(f"**/{ext}"))
        if crop_dir:
            crop_path = Path(crop_dir)
            if crop_path.exists():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                    paths += list(crop_path.glob(f"**/{ext}"))

        # 尝试从缓存加载
        cache_key = self._cache_key(paths)
        cached = self._load_cache(cache_key)
        if cached is not None:
            self._embeddings = cached
            self._image_paths = list(cached.keys())
            print(f"[CLIP] 从缓存加载 {len(self._embeddings)} 张图库 embedding")
            return len(self._embeddings)

        # 缓存未命中，重新计算
        self._embeddings.clear()
        for img_path in paths:
            try:
                pil = Image.open(img_path).convert("RGB")
                emb = self._encode(pil)
                self._embeddings[str(img_path)] = emb
            except Exception:
                continue

        self._image_paths = list(self._embeddings.keys())

        # 保存缓存
        self._save_cache(cache_key, self._embeddings)

        print(f"[CLIP] 预计算完成: {len(self._embeddings)} 张图库 embedding")
        return len(self._embeddings)

    def match_frame(self, frame_path: Path, frame_idx: int = 0) -> list[CLIPMatch]:
        """对单帧做 CLIP 语义匹配，返回 Top N 候选。"""
        if not self._embeddings:
            return []

        try:
            pil = Image.open(frame_path).convert("RGB")
            frame_emb = self._encode(pil)
        except Exception:
            return []

        # 批量余弦相似度
        paths = self._image_paths
        emb_matrix = np.stack([self._embeddings[p] for p in paths])
        similarities = emb_matrix @ frame_emb

        # 取 Top N
        top_indices = np.argsort(similarities)[::-1][:self.top_n]

        matches = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < self.threshold:
                break

            company_path = paths[idx]
            matches.append(CLIPMatch(
                frame_path=frame_path,
                frame_idx=frame_idx,
                company_path=company_path,
                company_name=Path(company_path).name,
                similarity=round(sim, 4),
                score=round(sim * 100, 1),
            ))

        return matches

    def match_frames(self, frame_paths: list[Path]) -> list[CLIPMatch]:
        """批量匹配多帧，返回所有候选（按分数排序）。"""
        all_matches = []
        for i, fp in enumerate(frame_paths):
            matches = self.match_frame(fp, frame_idx=i)
            all_matches.extend(matches)

        all_matches.sort(key=lambda m: m.similarity, reverse=True)

        # 跨帧去重：同一公司图只保留最高分
        seen = {}
        for m in all_matches:
            if m.company_path not in seen or m.similarity > seen[m.company_path].similarity:
                seen[m.company_path] = m

        return sorted(seen.values(), key=lambda m: m.similarity, reverse=True)

    def _encode(self, img: Image.Image) -> np.ndarray:
        """编码图片为 CLIP 向量（已归一化）。"""
        with torch.no_grad():
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            emb = self._model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    # ── 缓存 ──────────────────────────────────────────

    def _cache_key(self, paths: list[Path]) -> str:
        """根据图片路径列表生成缓存 key。"""
        path_str = "|".join(sorted(str(p) for p in paths))
        return hashlib.md5(path_str.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """缓存文件路径。"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"clip_{self.model_name}_{key}.pkl"

    def _load_cache(self, key: str) -> dict | None:
        """尝试从缓存加载 embeddings。"""
        cache_file = self._cache_path(key)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, key: str, embeddings: dict):
        """保存 embeddings 到缓存。"""
        try:
            cache_file = self._cache_path(key)
            with open(cache_file, "wb") as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            print(f"[CLIP] 缓存保存失败: {e}")
