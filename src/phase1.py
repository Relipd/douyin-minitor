"""Phase 1 — 三级漏斗初筛流程。

新架构替代原 suspect_detector + enhanced_detector 的复杂权重方案。

三级漏斗：
  Layer 1: 垃圾帧过滤（黑屏/白屏/熵/清晰度）
  Layer 2: SSIM 帧间去重
  Layer 3: CLIP 语义召回 + ORB 特征验证

设计目标：
  宁可多报，不要漏报，速度优先，人工最终判断。
"""
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field

from src.frame_filter import FrameFilter
from src.deduplicator import Deduplicator
from src.clip_matcher import CLIPMatcher, CLIPMatch
from src.orb_verifier import ORBVerifier, ORBVerifyResult


@dataclass
class Phase1Result:
    """Phase 1 单视频结果。"""
    video_id: str
    keyword: str
    total_frames: int = 0          # 截取总帧数
    junk_frames: int = 0           # 被过滤的垃圾帧数
    deduped_frames: int = 0        # 去重后帧数
    clip_candidates: int = 0       # CLIP 召回候选数
    verified_matches: int = 0      # ORB 验证通过数
    candidates: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)


class Phase1Pipeline:
    """三级漏斗初筛管线。"""

    def __init__(self, config: dict):
        """
        Args:
            config: config.yaml 中的完整配置
        """
        sc = config.get("screening", {})
        ff_cfg = config.get("frame_filter", {})
        dedup_cfg = config.get("dedup", {})
        clip_cfg = config.get("clip", {})
        orb_cfg = config.get("orb", {})

        self.frame_filter = FrameFilter(
            black_threshold=ff_cfg.get("black_threshold", 15.0),
            white_threshold=ff_cfg.get("white_threshold", 240.0),
            entropy_min=ff_cfg.get("entropy_min", 3.0),
            laplacian_min=ff_cfg.get("laplacian_min", 50.0),
        )

        self.deduplicator = Deduplicator(
            ssim_threshold=dedup_cfg.get("ssim_threshold", 0.92),
        )

        self.clip_matcher = CLIPMatcher(
            model_name=clip_cfg.get("model", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "laion2b_s34b_b79k"),
            threshold=clip_cfg.get("threshold", 0.25),
            top_n=clip_cfg.get("top_n", 5),
            cache_dir=clip_cfg.get("cache_dir", "./fingerprints"),
        )

        self.orb_verifier = ORBVerifier(
            n_features=orb_cfg.get("n_features", 2000),
            min_good_matches=orb_cfg.get("min_good_matches", 8),
            ratio_thresh=orb_cfg.get("ratio_thresh", 0.75),
            max_side=orb_cfg.get("max_side", 800),
        )

        self._company_full_dir = config.get("company_images_path", "./company_images")
        self._company_crop_dir = config.get("crop_images_path", "./crop_images")

    def load_library(self) -> int:
        """加载公司图库（CLIP embedding + ORB 特征）。"""
        # 收集所有公司图路径
        import os
        full_dir = self._resolve_path(self._company_full_dir)
        crop_dir = self._resolve_path(self._company_crop_dir)

        all_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            all_paths += [str(p) for p in Path(full_dir).glob(f"**/{ext}")]
        if crop_dir and Path(crop_dir).exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                all_paths += [str(p) for p in Path(crop_dir).glob(f"**/{ext}")]

        # CLIP 预计算
        n = self.clip_matcher.load_library(full_dir, crop_dir)

        # ORB 预计算
        self.orb_verifier.load_library(all_paths)

        return n

    def process_video(self, frame_dir: Path, video_id: str,
                      keyword: str = "") -> Phase1Result:
        """对一个视频的所有帧执行三级漏斗。

        Args:
            frame_dir: 截帧目录
            video_id: 视频 ID
            keyword: 搜索关键词

        Returns:
            Phase1Result
        """
        result = Phase1Result(video_id=video_id, keyword=keyword)

        # ── Layer 1: 垃圾帧过滤 ──
        frames = sorted([
            f for f in frame_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        result.total_frames = len(frames)

        valid_frames, quality_info = self.frame_filter.filter_with_info(frame_dir)
        result.junk_frames = result.total_frames - len(valid_frames)

        # ── Layer 2: SSIM 去重 ──
        deduped_frames, dedup_info = self.deduplicator.deduplicate_with_info(valid_frames)
        result.deduped_frames = len(deduped_frames)

        # ── Layer 3: CLIP 语义召回 + ORB 验证 ──
        clip_matches = self.clip_matcher.match_frames(deduped_frames)
        result.clip_candidates = len(clip_matches)

        # 构建验证候选
        verify_candidates = []
        for m in clip_matches:
            verify_candidates.append({
                "frame_path": m.frame_path,
                "company_path": m.company_path,
                "frame_idx": m.frame_idx,
                "clip_similarity": m.similarity,
            })

        # ORB 验证
        verified = self.orb_verifier.verify_candidates(verify_candidates)
        result.verified_matches = len(verified)

        # 合并结果
        for v in verified:
            # 找到对应的 CLIP 分数
            clip_score = 0.0
            for m in clip_matches:
                if m.company_path == v.company_path and m.frame_path == v.frame_path:
                    clip_score = m.similarity
                    break

            result.candidates.append({
                "frame_path": str(v.frame_path),
                "frame_idx": int(v.frame_path.stem.split("_")[1]) if "_" in v.frame_path.stem else 0,
                "company_path": v.company_path,
                "company_name": v.company_name,
                "clip_similarity": clip_score,
                "orb_good_matches": v.good_matches,
                "orb_match_rate": v.match_rate,
                "orb_score": v.score,
                # 综合分 = CLIP 语义 40% + ORB 特征验证 60%
                # CLIP: 0~1 → 0~40分, ORB: 0~100 → 0~60分
                "final_score": round(clip_score * 40 + v.score * 0.6, 1),
            })

        # 按综合分排序
        result.candidates.sort(key=lambda c: c["final_score"], reverse=True)

        return result

    def save_evidence(self, result: Phase1Result, output_dir: Path) -> Path | None:
        """保存疑似证据包。"""
        if not result.candidates:
            return None

        review_dir = output_dir / result.video_id
        review_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for i, c in enumerate(result.candidates):
            frame_path = Path(c["frame_path"])
            fname = f"frame_{c['frame_idx']:04d}__vs__{c['company_name']}.jpg"
            shutil.copy2(frame_path, review_dir / fname)
            saved.append((c, fname))

        # 生成配对对比图
        self._generate_pairs(review_dir, saved, result)

        # 生成 summary.json
        summary = {
            "video_id": result.video_id,
            "keyword": result.keyword,
            "total_frames": result.total_frames,
            "junk_frames": result.junk_frames,
            "deduped_frames": result.deduped_frames,
            "clip_candidates": result.clip_candidates,
            "verified_matches": result.verified_matches,
            "candidates": result.candidates,
        }
        with open(review_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 生成 HTML
        self._write_html(review_dir, saved, result)

        return review_dir

    def _generate_pairs(self, review_dir: Path, saved: list, result: Phase1Result):
        """生成配对对比图。"""
        from PIL import Image, ImageDraw, ImageFont

        full_dir = self._resolve_path(self._company_full_dir)
        crop_dir = self._resolve_path(self._company_crop_dir)

        pairs = []
        for i, (candidate, fname) in enumerate(saved):
            frame_path = Path(candidate["frame_path"])
            company_name = candidate["company_name"]

            # 查找公司图
            company_path = self._find_company_image(company_name, full_dir, crop_dir)
            if company_path is None:
                continue

            # 生成对比图
            pair_name = f"pair_{i+1:03d}.jpg"
            pair_path = review_dir / pair_name
            self._create_pair_image(frame_path, company_path, pair_path)

            pairs.append({
                "index": i + 1,
                "pair_image": pair_name,
                "frame_file": fname,
                "company_file": company_name,
                "clip_similarity": candidate.get("clip_similarity", 0),
                "orb_good_matches": candidate.get("orb_good_matches", 0),
                "screening_score": candidate.get("final_score", 0),
            })

        # 保存 pairs.json
        meta = {
            "video_id": result.video_id,
            "keyword": result.keyword,
            "total_pairs": len(pairs),
            "pairs": pairs,
        }
        with open(review_dir / "pairs.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _find_company_image(self, name: str, full_dir: str, crop_dir: str) -> Path | None:
        """根据文件名查找公司图片。"""
        for d in [full_dir, crop_dir]:
            if not d or not Path(d).exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for p in Path(d).glob(f"**/{ext}"):
                    if p.name == name or p.stem == name:
                        return p
        return None

    def _create_pair_image(self, frame_path: Path, company_path: Path,
                           output_path: Path, target_h: int = 500):
        """创建左右对比图。"""
        from PIL import Image, ImageDraw, ImageFont

        try:
            frame_img = Image.open(frame_path).convert("RGB")
        except Exception:
            frame_img = Image.new("RGB", (640, 480), (50, 50, 50))

        try:
            company_img = Image.open(str(company_path)).convert("RGB")
        except Exception:
            company_img = Image.new("RGB", (300, 300), (50, 50, 50))

        # 统一高度
        fw, fh = frame_img.size
        cw, ch = company_img.size
        frame_resized = frame_img.resize(
            (int(fw * target_h / fh), target_h), Image.LANCZOS
        )
        company_resized = company_img.resize(
            (int(cw * target_h / ch), target_h), Image.LANCZOS
        )

        gap = 10
        label_h = 40
        total_w = frame_resized.width + gap + company_resized.width
        total_h = target_h + label_h

        canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        canvas.paste(frame_resized, (0, label_h))
        canvas.paste(company_resized, (frame_resized.width + gap, label_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("msyh.ttc", 18)
        except Exception:
            font = ImageFont.load_default()

        draw.text((10, 8), "VIDEO FRAME", fill=(100, 200, 255), font=font)
        draw.text((frame_resized.width + gap + 10, 8), "COMPANY IMAGE",
                  fill=(255, 200, 100), font=font)

        canvas.save(str(output_path), quality=95)

    def _write_html(self, review_dir: Path, saved: list, result: Phase1Result):
        """生成 HTML 报告。"""
        rows = ""
        for c, fname in saved:
            clip_tag = f"{c.get('clip_similarity', 0):.2f}"
            orb_tag = f"{c.get('orb_good_matches', 0)} pts"

            rows += f"""
            <tr>
                <td><img src="{fname}" class="thumb" onclick="this.classList.toggle('zoomed')"></td>
                <td>{c['company_name']}</td>
                <td class="score">{c.get('final_score', 0):.0f}</td>
                <td>CLIP: {clip_tag}</td>
                <td>ORB: {orb_tag}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>Phase1 筛查 - {result.video_id}</title>
<style>
    body{{font-family:'Microsoft YaHei',sans-serif;margin:20px;background:#111;color:#eee}}
    h2{{color:#4fc3f7}}
    .info{{color:#aaa;margin-bottom:20px}}
    .summary{{display:flex;gap:20px;margin-bottom:20px}}
    .stat{{background:#1a1a2e;padding:15px;border-radius:8px;text-align:center}}
    .stat .num{{font-size:24px;font-weight:bold;color:#4fc3f7}}
    .stat .label{{font-size:12px;color:#aaa;margin-top:4px}}
    table{{border-collapse:collapse;width:100%}}
    th{{background:#333;padding:10px;position:sticky;top:0}}
    td{{padding:8px;text-align:center;border-bottom:1px solid #333}}
    .score{{font-weight:bold;font-size:18px;color:#ff9800}}
    .thumb{{max-height:300px;cursor:pointer;transition:all 0.3s}}
    .zoomed{{max-height:90vh;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:999}}
    tr:hover{{background:#222}}
</style></head><body>
<h2>Phase 1 — 三级漏斗筛选</h2>
<div class="info">视频: {result.video_id} | 关键词: {result.keyword}</div>
<div class="summary">
    <div class="stat"><div class="num">{result.total_frames}</div><div class="label">总帧数</div></div>
    <div class="stat"><div class="num">{result.junk_frames}</div><div class="label">垃圾帧过滤</div></div>
    <div class="stat"><div class="num">{result.deduped_frames}</div><div class="label">去重后</div></div>
    <div class="stat"><div class="num">{result.clip_candidates}</div><div class="label">CLIP 召回</div></div>
    <div class="stat"><div class="num">{result.verified_matches}</div><div class="label">验证通过</div></div>
</div>
<table><thead><tr>
    <th>帧</th><th>公司图</th><th>综合分</th><th>CLIP</th><th>ORB</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
        (review_dir / "index.html").write_text(html, encoding="utf-8")

    def _resolve_path(self, path_str: str) -> str:
        """解析相对路径为绝对路径。"""
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str(Path.cwd() / p)
