"""配对对比图生成器。

将 Python 初筛产出的疑似帧 + 对应公司图 生成并排对比图，
供 AI 二次细筛使用。

支持两种输入格式：
  1. SuspectDetector 产出的 review 目录（含 summary.json + 帧文件）
  2. screening.py 产出的 review 目录（已含 pairs.json）
"""
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


@dataclass
class PairEntry:
    """一个配对对比条目。"""
    index: int
    frame_file: str
    company_file: str
    pair_image: str
    phash_distance: int = 0
    dhash_distance: int = 0
    good_matches: int = 0
    inliers: int = 0
    screening_score: float = 0.0


class PairGenerator:
    """从 review 目录生成配对对比图。"""

    def __init__(self, company_images_dir: str, crop_images_dir: str = None,
                 target_h: int = 500, frame_width: int = 1920):
        """
        Args:
            company_images_dir: 公司图片目录（用于定位原始图）。
            crop_images_dir: 裁剪图目录（可选）。
            target_h: 对比图高度。
            frame_width: 视频帧宽度（用于缩放）。
        """
        self.company_images_dir = Path(company_images_dir)
        self.crop_images_dir = Path(crop_images_dir) if crop_images_dir else None
        self.target_h = target_h
        self.frame_width = frame_width

        # 公司图名称 → 路径映射
        self._image_index = {}
        self._build_index()

    def _build_index(self):
        """建立公司图文件名到路径的映射。"""
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for p in self.company_images_dir.glob(f"**/{ext}"):
                self._image_index[p.name] = p
                self._image_index[p.stem] = p
            if self.crop_images_dir and self.crop_images_dir.exists():
                for p in self.crop_images_dir.glob(f"**/{ext}"):
                    self._image_index[p.name] = p
                    self._image_index[p.stem] = p

    def find_company_image(self, name: str) -> Path | None:
        """根据文件名查找公司图片路径。"""
        # 精确匹配
        if name in self._image_index:
            return self._image_index[name]
        # 去扩展名匹配
        stem = Path(name).stem
        if stem in self._image_index:
            return self._image_index[stem]
        # 模糊匹配
        for key, path in self._image_index.items():
            if name in key or key in name:
                return path
        return None

    def generate_from_suspect_detector(self, review_dir: Path) -> Path:
        """从 SuspectDetector 产出的 review 目录生成配对图。

        SuspectDetector 的目录结构：
          review_dir/
            frame_0012__vs__公司图A.jpg   (仅视频帧副本)
            summary.json
            index.html
        """
        review_dir = Path(review_dir)
        summary_file = review_dir / "summary.json"

        if not summary_file.exists():
            raise FileNotFoundError(f"summary.json not found in {review_dir}")

        with open(summary_file, encoding="utf-8") as f:
            summary = json.load(f)

        pairs = []
        for i, frame_info in enumerate(summary.get("frames", [])):
            company_name = frame_info.get("company", "")
            frame_idx = frame_info.get("frame_idx", 0)
            score = frame_info.get("score", 0)

            # 查找帧文件
            frame_file = review_dir / f"frame_{frame_idx+1:04d}__vs__{company_name}"
            if not frame_file.exists():
                # 尝试其他命名模式
                candidates = list(review_dir.glob(f"frame_{frame_idx+1:04d}*{company_name}*"))
                if candidates:
                    frame_file = candidates[0]
                else:
                    # 尝试模糊匹配
                    candidates = list(review_dir.glob(f"frame_*__vs__{company_name}*"))
                    if candidates:
                        frame_file = candidates[0]
                    else:
                        continue

            # 查找公司原图
            company_path = self.find_company_image(company_name)
            if company_path is None:
                continue

            # 生成配对图
            pair_name = f"pair_{i+1:03d}.jpg"
            pair_path = review_dir / pair_name
            self._create_pair_image(frame_file, company_path, pair_path)

            pairs.append({
                "index": i + 1,
                "pair_image": pair_name,
                "frame_file": frame_file.name,
                "company_file": company_name,
                "phash_distance": frame_info.get("phash_distance", 0),
                "dhash_distance": frame_info.get("dhash_distance", 0),
                "good_matches": frame_info.get("good_matches", 0),
                "inliers": frame_info.get("inliers", 0),
                "screening_score": score,
            })

        # 保存 pairs.json
        meta = {
            "video_id": summary.get("video_id", review_dir.name),
            "video_url": summary.get("video_url", ""),
            "keyword": summary.get("keyword", ""),
            "total_pairs": len(pairs),
            "pairs": pairs,
        }
        with open(review_dir / "pairs.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return review_dir

    def generate_from_review_dirs(self, review_base_dir: Path) -> list[Path]:
        """批量处理 review 基目录下的所有子目录。"""
        review_base_dir = Path(review_base_dir)
        processed = []

        for subdir in sorted(review_base_dir.iterdir()):
            if not subdir.is_dir():
                continue
            try:
                self.generate_from_suspect_detector(subdir)
                processed.append(subdir)
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"  [WARN] {subdir.name}: {e}")

        return processed

    def _create_pair_image(self, frame_path: Path, company_path: Path,
                           output_path: Path):
        """创建左=视频帧、右=公司图的并排对比图。"""
        try:
            frame_img = Image.open(frame_path).convert("RGB")
        except Exception:
            frame_img = Image.new("RGB", (640, 480), (50, 50, 50))
            draw = ImageDraw.Draw(frame_img)
            draw.text((200, 220), "Frame Not Found", fill=(255, 100, 100))

        try:
            company_img = Image.open(str(company_path)).convert("RGB")
        except Exception:
            company_img = Image.new("RGB", (300, 300), (50, 50, 50))
            draw = ImageDraw.Draw(company_img)
            draw.text((80, 140), "Image Not Found", fill=(255, 100, 100))

        # 统一高度
        fw, fh = frame_img.size
        cw, ch = company_img.size
        frame_resized = frame_img.resize(
            (int(fw * self.target_h / fh), self.target_h), Image.LANCZOS
        )
        company_resized = company_img.resize(
            (int(cw * self.target_h / ch), self.target_h), Image.LANCZOS
        )

        gap = 10
        label_h = 40
        total_w = frame_resized.width + gap + company_resized.width
        total_h = self.target_h + label_h

        canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        canvas.paste(frame_resized, (0, label_h))
        canvas.paste(company_resized, (frame_resized.width + gap, label_h))

        # 标签
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("msyh.ttc", 18)
        except Exception:
            font = ImageFont.load_default()

        draw.text((10, 8), "VIDEO FRAME", fill=(100, 200, 255), font=font)
        draw.text((frame_resized.width + gap + 10, 8), "COMPANY IMAGE",
                  fill=(255, 200, 100), font=font)

        canvas.save(str(output_path), quality=95)
