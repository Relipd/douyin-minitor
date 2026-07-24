"""垃圾帧过滤模块 — Phase 1 第一层漏斗。

目标：从截取的帧中快速剔除无价值帧，减少后续匹配计算量。

过滤规则：
  1. 黑屏检测：平均亮度 < black_threshold
  2. 白屏检测：平均亮度 > white_threshold
  3. 信息熵检测：熵值 < entropy_min（纯色/无内容）
  4. 清晰度检测：Laplacian 方差 < laplacian_min（模糊帧）
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FrameQuality:
    """单帧质量指标。"""
    frame_path: Path
    frame_idx: int
    mean_brightness: float
    entropy: float
    laplacian_var: float
    is_junk: bool
    junk_reason: str = ""


class FrameFilter:
    """垃圾帧过滤器。

    设计目标：快速、高精度，宁可多留不可多删。
    """

    def __init__(self,
                 black_threshold: float = 15.0,
                 white_threshold: float = 240.0,
                 entropy_min: float = 3.0,
                 laplacian_min: float = 50.0):
        """
        Args:
            black_threshold: 亮度低于此值判定为黑屏
            white_threshold: 亮度高于此值判定为白屏
            entropy_min: 信息熵低于此值判定为无内容
            laplacian_min: Laplacian方差低于此值判定为模糊
        """
        self.black_threshold = black_threshold
        self.white_threshold = white_threshold
        self.entropy_min = entropy_min
        self.laplacian_min = laplacian_min

    def analyze_frame(self, frame_path: Path, frame_idx: int = 0) -> FrameQuality:
        """分析单帧质量指标。"""
        img = cv2.imread(str(frame_path))
        if img is None:
            return FrameQuality(
                frame_path=frame_path,
                frame_idx=frame_idx,
                mean_brightness=0,
                entropy=0,
                laplacian_var=0,
                is_junk=True,
                junk_reason="无法读取",
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_brightness = float(gray.mean())
        entropy = self._shannon_entropy(gray)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        is_junk, reason = self._check_junk(mean_brightness, entropy, laplacian_var)

        return FrameQuality(
            frame_path=frame_path,
            frame_idx=frame_idx,
            mean_brightness=round(mean_brightness, 2),
            entropy=round(entropy, 4),
            laplacian_var=round(laplacian_var, 2),
            is_junk=is_junk,
            junk_reason=reason,
        )

    def filter_frames(self, frame_dir: Path) -> list[Path]:
        """过滤目录下所有帧，返回有效帧路径列表。"""
        frames = sorted([
            f for f in frame_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        valid = []
        for i, fp in enumerate(frames):
            quality = self.analyze_frame(fp, frame_idx=i)
            if not quality.is_junk:
                valid.append(fp)

        return valid

    def filter_with_info(self, frame_dir: Path) -> tuple[list[Path], list[FrameQuality]]:
        """过滤并返回详细质量信息。"""
        frames = sorted([
            f for f in frame_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        valid = []
        all_quality = []
        for i, fp in enumerate(frames):
            quality = self.analyze_frame(fp, frame_idx=i)
            all_quality.append(quality)
            if not quality.is_junk:
                valid.append(fp)

        return valid, all_quality

    def _check_junk(self, mean_brightness: float, entropy: float,
                    laplacian_var: float) -> tuple[bool, str]:
        """判断是否为垃圾帧。"""
        if mean_brightness < self.black_threshold:
            return True, f"黑屏(亮度={mean_brightness:.1f})"
        if mean_brightness > self.white_threshold:
            return True, f"白屏(亮度={mean_brightness:.1f})"
        if entropy < self.entropy_min:
            return True, f"无内容(熵={entropy:.2f})"
        if laplacian_var < self.laplacian_min:
            return True, f"模糊(清晰度={laplacian_var:.1f})"
        return False, ""

    @staticmethod
    def _shannon_entropy(gray: np.ndarray) -> float:
        """计算灰度图的 Shannon 信息熵。"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
