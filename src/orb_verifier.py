"""ORB 特征验证模块 — Phase 1 第三层漏斗验证环节。

目标：对 CLIP 召回的候选做 ORB 特征点匹配验证，
      确保候选帧与版权图在像素级也有对应关系。

使用场景：
  CLIP 做语义召回（"看起来像同一个东西"）
  ORB 做特征验证（"细节也能对上"）
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    from PIL import Image
except ImportError:
    pass


@dataclass
class ORBVerifyResult:
    """ORB 验证结果。"""
    frame_path: Path
    company_path: str
    company_name: str
    good_matches: int
    match_rate: float
    verified: bool
    score: float  # 综合验证分


class ORBVerifier:
    """ORB 特征验证器。

    流程：
      1. 对视频帧提取 ORB 特征
      2. 对公司图提取 ORB 特征
      3. BFMatcher + Lowe's ratio test
      4. good_matches >= threshold → 验证通过
    """

    def __init__(self,
                 n_features: int = 2000,
                 min_good_matches: int = 8,
                 ratio_thresh: float = 0.75,
                 max_side: int = 800):
        """
        Args:
            n_features: ORB 最大特征点数
            min_good_matches: 最少匹配点数，低于此值验证不通过
            ratio_thresh: Lowe's ratio test 阈值
            max_side: 图片缩放最大边长
        """
        self.n_features = n_features
        self.min_good_matches = min_good_matches
        self.ratio_thresh = ratio_thresh
        self.max_side = max_side

        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 公司图预计算 {path: (keypoints, descriptors, image)}
        self._company_features: dict[str, tuple] = {}

    def load_library(self, company_image_paths: list[str]):
        """预计算公司图库的 ORB 特征。"""
        self._company_features.clear()

        for img_path in company_image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 缩放
            h, w = img.shape[:2]
            if max(w, h) > self.max_side:
                scale = self.max_side / max(w, h)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self._orb.detectAndCompute(gray, None)

            if des is not None and len(kp) >= 5:
                self._company_features[img_path] = (kp, des, gray)

        print(f"[ORB] 预计算完成: {len(self._company_features)} 张图库特征")

    def verify(self, frame_path: Path, company_path: str,
               frame_idx: int = 0) -> ORBVerifyResult:
        """验证单帧与单张公司图的 ORB 匹配。"""
        frame_img = cv2.imread(str(frame_path))
        if frame_img is None:
            return self._no_match(frame_path, company_path, frame_idx)

        # 缩放
        h, w = frame_img.shape[:2]
        if max(w, h) > self.max_side:
            scale = self.max_side / max(w, h)
            frame_img = cv2.resize(frame_img, (int(w * scale), int(h * scale)))

        frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self._orb.detectAndCompute(frame_gray, None)

        if des1 is None or len(kp1) < 5:
            return self._no_match(frame_path, company_path, frame_idx)

        # 获取公司图特征
        if company_path not in self._company_features:
            return self._no_match(frame_path, company_path, frame_idx)

        kp2, des2, _ = self._company_features[company_path]

        # BFMatcher + ratio test
        try:
            raw_matches = self._bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            return self._no_match(frame_path, company_path, frame_idx)

        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        match_rate = len(good_matches) / max(min(len(des1), len(des2)), 1)
        verified = len(good_matches) >= self.min_good_matches

        # ORB 综合分 (0~100)：基于绝对匹配点数，不受特征点总数稀释
        # 每个 good match 贡献约 8 分，达到 min_good_matches 阈值时 ~48 分起
        score = min(100, round(len(good_matches) * 8, 1))

        return ORBVerifyResult(
            frame_path=frame_path,
            company_path=company_path,
            company_name=Path(company_path).name,
            good_matches=len(good_matches),
            match_rate=round(match_rate, 4),
            verified=verified,
            score=round(score, 1),
        )

    def verify_candidates(self, candidates: list[dict]) -> list[ORBVerifyResult]:
        """批量验证候选列表。

        Args:
            candidates: [{"frame_path": Path, "company_path": str, "frame_idx": int}]

        Returns:
            验证通过的候选列表
        """
        results = []
        for c in candidates:
            result = self.verify(
                frame_path=c["frame_path"],
                company_path=c["company_path"],
                frame_idx=c.get("frame_idx", 0),
            )
            if result.verified:
                results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _no_match(self, frame_path: Path, company_path: str,
                  frame_idx: int) -> ORBVerifyResult:
        """返回未匹配结果。"""
        return ORBVerifyResult(
            frame_path=frame_path,
            company_path=company_path,
            company_name=Path(company_path).name,
            good_matches=0,
            match_rate=0.0,
            verified=False,
            score=0.0,
        )
