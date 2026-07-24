"""SuspectDetector 单元测试。

测试内容:
  1. 图库加载
  2. 单帧筛查
  3. 批量筛查
  4. 去重逻辑
  5. 连续帧加分
  6. 证据保存
"""
import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacy.suspect_detector import SuspectDetector, SuspectMatch


def _make_test_images(tmp_dir):
    """创建临时测试图片。

    返回 (company_dir, frame_dir, company_path, frame_path, diff_frame_path)
    """
    company_dir = Path(tmp_dir) / "company"
    frame_dir = Path(tmp_dir) / "frames"
    company_dir.mkdir()
    frame_dir.mkdir()

    # 创建一张红色产品图（模拟公司图）
    arr1 = np.zeros((200, 200, 3), dtype=np.uint8)
    arr1[:, :, 2] = 200  # 红色
    arr1[50:150, 50:150] = [255, 100, 50]  # 中心橙色块
    img1 = Image.fromarray(arr1)
    company_path = company_dir / "product_red.png"
    img1.save(str(company_path))

    # 创建一张相似的帧图（模拟视频帧）
    arr2 = np.zeros((300, 400, 3), dtype=np.uint8)
    arr2[:, :, 2] = 190  # 略有不同的红色
    arr2[80:180, 100:250] = [250, 110, 60]  # 略偏移的橙色块
    img2 = Image.fromarray(arr2)
    frame_path = frame_dir / "frame_0001.jpg"
    img2.save(str(frame_path), quality=85)

    # 创建一张完全不同的帧图（蓝色）
    arr3 = np.zeros((300, 400, 3), dtype=np.uint8)
    arr3[:, :, 0] = 200  # 蓝色
    img3 = Image.fromarray(arr3)
    diff_frame_path = frame_dir / "frame_0002.jpg"
    img3.save(str(diff_frame_path), quality=85)

    return company_dir, frame_dir, company_path, frame_path, diff_frame_path


class TestLoadLibrary(unittest.TestCase):
    """图库加载测试。"""

    def test_load_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = SuspectDetector()
            n = detector.load_library(tmp)
            self.assertEqual(n, 0)

    def test_load_with_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, _, _, _, _ = _make_test_images(tmp)
            detector = SuspectDetector()
            n = detector.load_library(str(company_dir))
            self.assertEqual(n, 1)

    def test_load_with_both_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, _, _, _, _ = _make_test_images(tmp)
            crop_dir = Path(tmp) / "crop"
            crop_dir.mkdir()
            src = list(company_dir.glob("*.png"))[0]
            shutil.copy2(str(src), str(crop_dir / "crop_product.png"))

            detector = SuspectDetector()
            n = detector.load_library(str(company_dir), str(crop_dir))
            self.assertEqual(n, 2)


class TestScreenFrame(unittest.TestCase):
    """单帧筛查测试。"""

    def test_screen_similar_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, _, company_path, frame_path, _ = _make_test_images(tmp)
            detector = SuspectDetector(max_side=400, phash_max=64, min_score=0)
            detector.load_library(str(company_dir))

            matches = detector.screen_frame(frame_path, frame_idx=0)
            self.assertIsInstance(matches, list)

    def test_screen_different_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, _, _, _, diff_frame = _make_test_images(tmp)
            detector = SuspectDetector(max_side=400, phash_max=30, min_score=0)
            detector.load_library(str(company_dir))

            matches = detector.screen_frame(diff_frame, frame_idx=0)
            self.assertEqual(len(matches), 0)

    def test_screen_nonexistent_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, _, _, _, _ = _make_test_images(tmp)
            detector = SuspectDetector()
            detector.load_library(str(company_dir))

            matches = detector.screen_frame(Path("/nonexistent/frame.jpg"))
            self.assertEqual(matches, [])

    def test_match_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, _, company_path, frame_path, _ = _make_test_images(tmp)
            detector = SuspectDetector(max_side=400, phash_max=64, min_score=0)
            detector.load_library(str(company_dir))

            matches = detector.screen_frame(frame_path, frame_idx=5)
            if matches:
                m = matches[0]
                self.assertIsInstance(m, SuspectMatch)
                self.assertEqual(m.frame_idx, 5)
                self.assertEqual(m.frame_path, frame_path)
                self.assertGreaterEqual(m.hist_score, 0)
                self.assertLessEqual(m.hist_score, 1.0)
                self.assertGreaterEqual(m.phash_distance, 0)
                self.assertGreaterEqual(m.dhash_distance, 0)
                self.assertEqual(m.consecutive_hits, 1)


class TestScreenFrames(unittest.TestCase):
    """批量筛查测试。"""

    def test_screen_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = SuspectDetector()
            suspects = detector.screen_frames(Path(tmp))
            self.assertEqual(suspects, [])

    def test_screen_with_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, frame_dir, _, _, _ = _make_test_images(tmp)
            detector = SuspectDetector(max_side=400, phash_max=64, min_score=0)
            detector.load_library(str(company_dir))

            suspects = detector.screen_frames(frame_dir)
            self.assertIsInstance(suspects, list)

    def test_suspects_sorted_by_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, frame_dir, _, _, _ = _make_test_images(tmp)
            detector = SuspectDetector(max_side=400, phash_max=64, min_score=0)
            detector.load_library(str(company_dir))

            suspects = detector.screen_frames(frame_dir)
            if len(suspects) >= 2:
                scores = [s.final_score for s in suspects]
                self.assertEqual(scores, sorted(scores, reverse=True))


class TestDeduplication(unittest.TestCase):
    """去重逻辑测试。"""

    def test_dedup_removes_nearby_frames(self):
        detector = SuspectDetector(dedup_gap=5)
        company_img = "/test/product.jpg"

        matches = [
            SuspectMatch(
                frame_path=Path(f"/test/frame_{i}.jpg"),
                frame_idx=i,
                company_image=company_img,
                company_image_name="product.jpg",
                phash_distance=10, dhash_distance=10,
                hist_score=0.8, ssim_score=0.0,
                final_score=70.0 + i,
                ocr_keywords=[], consecutive_hits=1,
            )
            for i in range(10)
        ]

        deduped = detector._deduplicate(matches)
        self.assertLess(len(deduped), len(matches))

    def test_dedup_keeps_best_plus_nearby(self):
        """去重保留最高分帧 + dedup_gap 范围内的帧。"""
        detector = SuspectDetector(dedup_gap=5)
        company_img = "/test/product.jpg"

        # 5 个连续帧，分数递增，gap=5 应保留多个
        matches = [
            SuspectMatch(
                frame_path=Path(f"/test/frame_{i}.jpg"),
                frame_idx=i,
                company_image=company_img,
                company_image_name="product.jpg",
                phash_distance=10, dhash_distance=10,
                hist_score=0.8, ssim_score=0.0,
                final_score=50.0 + i * 2,
                ocr_keywords=[], consecutive_hits=1,
            )
            for i in range(10)
        ]

        deduped = detector._deduplicate(matches)
        # 最高分在 index=9, score=68; gap=5 → 保留 index 4~9 (6帧)
        self.assertGreater(len(deduped), 1)
        self.assertLessEqual(len(deduped), 6)

    def test_dedup_different_companies(self):
        detector = SuspectDetector(dedup_gap=5)

        matches = [
            SuspectMatch(
                frame_path=Path(f"/test/frame_{i}.jpg"),
                frame_idx=i,
                company_image=f"/test/company_{i % 2}.jpg",
                company_image_name=f"company_{i % 2}.jpg",
                phash_distance=10, dhash_distance=10,
                hist_score=0.8, ssim_score=0.0,
                final_score=60.0,
                ocr_keywords=[], consecutive_hits=1,
            )
            for i in range(6)
        ]

        deduped = detector._deduplicate(matches)
        self.assertGreaterEqual(len(deduped), 2)


class TestConsecutiveBonus(unittest.TestCase):
    """连续帧加分测试。"""

    def test_consecutive_bonus_applied(self):
        detector = SuspectDetector(consecutive_bonus=15)
        company_img = "/test/product.jpg"

        matches = [
            SuspectMatch(
                frame_path=Path(f"/test/frame_{i}.jpg"),
                frame_idx=i,
                company_image=company_img,
                company_image_name="product.jpg",
                phash_distance=10, dhash_distance=10,
                hist_score=0.8, ssim_score=0.0,
                final_score=50.0,
                ocr_keywords=[], consecutive_hits=1,
            )
            for i in range(5)
        ]

        result = detector._apply_consecutive_bonus(matches)
        has_consecutive = any(m.consecutive_hits > 1 for m in result)
        self.assertTrue(has_consecutive)

    def test_no_bonus_for_isolated_frames(self):
        detector = SuspectDetector(consecutive_bonus=15)
        company_img = "/test/product.jpg"

        matches = [
            SuspectMatch(
                frame_path=Path(f"/test/frame_{i*100}.jpg"),
                frame_idx=i * 100,
                company_image=company_img,
                company_image_name="product.jpg",
                phash_distance=10, dhash_distance=10,
                hist_score=0.8, ssim_score=0.0,
                final_score=50.0,
                ocr_keywords=[], consecutive_hits=1,
            )
            for i in range(3)
        ]

        result = detector._apply_consecutive_bonus(matches)
        for m in result:
            self.assertEqual(m.consecutive_hits, 1)


class TestSaveEvidence(unittest.TestCase):
    """证据保存测试。"""

    def test_save_evidence_creates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, frame_dir, _, frame_path, _ = _make_test_images(tmp)
            detector = SuspectDetector(max_side=400, phash_max=64, min_score=0)
            detector.load_library(str(company_dir))

            suspects = detector.screen_frame(frame_path, frame_idx=0)
            if suspects:
                output_dir = Path(tmp) / "review"
                output_dir.mkdir()
                out = detector.save_evidence(
                    suspects, output_dir,
                    video_id="test_123", keyword="test",
                )
                self.assertIsNotNone(out)
                self.assertTrue(out.exists())
                self.assertTrue((out / "summary.json").exists())
                self.assertTrue((out / "index.html").exists())

    def test_save_evidence_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = SuspectDetector()
            output_dir = Path(tmp) / "review"
            output_dir.mkdir()
            result = detector.save_evidence([], output_dir)
            self.assertIsNone(result)


class TestScoreCalculation(unittest.TestCase):
    """评分计算测试。"""

    def test_score_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, _, _, frame_path, _ = _make_test_images(tmp)
            detector = SuspectDetector(max_side=400, phash_max=64, min_score=0)
            detector.load_library(str(company_dir))

            matches = detector.screen_frame(frame_path, frame_idx=0)
            for m in matches:
                self.assertGreaterEqual(m.final_score, 0)
                self.assertLessEqual(m.final_score, 100)

    def test_min_score_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            company_dir, frame_dir, _, _, _ = _make_test_images(tmp)
            detector = SuspectDetector(
                max_side=400, phash_max=64, min_score=999
            )
            detector.load_library(str(company_dir))

            suspects = detector.screen_frames(frame_dir)
            self.assertEqual(len(suspects), 0)


if __name__ == "__main__":
    unittest.main()
