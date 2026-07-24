"""帧间去重模块 — Phase 1 第二层漏斗。

目标：相邻帧高度相似时只保留一帧，减少重复匹配计算。

使用 SSIM（结构相似性）判断帧间重复。
"""
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


class Deduplicator:
    """帧间去重器。

    设计目标：
      用 SSIM 检测相邻帧的结构相似性，
      相似度超过阈值的帧直接丢弃，只保留一帧。
    """

    def __init__(self, ssim_threshold: float = 0.92):
        """
        Args:
            ssim_threshold: SSIM 高于此值判定为重复帧（0~1）
        """
        self.ssim_threshold = ssim_threshold

    def deduplicate(self, frame_paths: list[Path]) -> list[Path]:
        """对帧列表去重，返回去重后的帧路径列表。

        Args:
            frame_paths: 已排序的帧路径列表

        Returns:
            去重后的帧路径列表
        """
        if len(frame_paths) <= 1:
            return frame_paths

        kept = [frame_paths[0]]
        prev_gray = self._load_gray(frame_paths[0])

        for fp in frame_paths[1:]:
            curr_gray = self._load_gray(fp)
            if curr_gray is None:
                continue

            if prev_gray is not None:
                similarity = self._calc_ssim(prev_gray, curr_gray)
                if similarity >= self.ssim_threshold:
                    continue  # 重复帧，跳过

            kept.append(fp)
            prev_gray = curr_gray

        return kept

    def deduplicate_with_info(self, frame_paths: list[Path]) -> tuple[list[Path], list[dict]]:
        """去重并返回每帧的 SSIM 信息。"""
        if len(frame_paths) <= 1:
            return frame_paths, []

        kept = [frame_paths[0]]
        prev_gray = self._load_gray(frame_paths[0])
        info = [{"frame": frame_paths[0].name, "ssim_with_prev": 1.0, "kept": True}]

        for fp in frame_paths[1:]:
            curr_gray = self._load_gray(fp)
            if curr_gray is None:
                info.append({"frame": fp.name, "ssim_with_prev": 0, "kept": False, "reason": "无法读取"})
                continue

            similarity = 0.0
            if prev_gray is not None:
                similarity = self._calc_ssim(prev_gray, curr_gray)

            if similarity >= self.ssim_threshold:
                info.append({"frame": fp.name, "ssim_with_prev": round(similarity, 4), "kept": False})
            else:
                kept.append(fp)
                prev_gray = curr_gray
                info.append({"frame": fp.name, "ssim_with_prev": round(similarity, 4), "kept": True})

        return kept, info

    @staticmethod
    def _load_gray(frame_path: Path) -> np.ndarray | None:
        """加载图片并转为灰度图。"""
        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _calc_ssim(gray1: np.ndarray, gray2: np.ndarray) -> float:
        """计算两张灰度图的 SSIM。"""
        # 确保尺寸一致
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        score, _ = ssim(gray1, gray2, full=True)
        return float(score)
