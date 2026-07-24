"""抖音版权监控 — Phase 1 三级漏斗初筛工具（直链模式）。

流程：
  Layer 1: 垃圾帧过滤（黑屏/白屏/熵/清晰度）
  Layer 2: SSIM 帧间去重
  Layer 3: CLIP 语义召回 + ORB 特征验证

用法:
    python main.py --url https://www.douyin.com/video/xxx          # 单视频直链
    python main.py --url url1 url2 url3                             # 多个视频
    python main.py --url-file urls.txt                              # 从文件读链接
    python main.py --excel 链接.xlsx                                 # 从 Excel B 列读链接
    python main.py --headless                                       # 无头模式
"""
import argparse
import asyncio
import json
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from contextlib import contextmanager

from src.crawler import DouyinCrawler
from src.phase1 import Phase1Pipeline
from src.utils import load_config as _load_config, resolve_path as _resolve_path

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path=None):
    return _load_config(path, base_dir=BASE_DIR)


def resolve_path(cfg, key, default=None):
    return _resolve_path(cfg, key, base_dir=BASE_DIR, default=default)


@contextmanager
def suppress_stderr():
    """静默 stderr（用于屏蔽 Playwright/Chromium 的噪音日志）。"""
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        devnull.close()
        sys.stderr = old_stderr


def extract_video_id(url: str) -> str:
    """从抖音 URL 中提取视频 ID。"""
    if "/video/" in url:
        return url.split("/video/")[-1].split("?")[0].rstrip("/")
    return url.rstrip("/").split("/")[-1].split("?")[0]


def read_urls_from_excel(filepath: str) -> list[str]:
    """从 Excel 文件的 B 列读取视频链接。"""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] Excel 文件不存在: {filepath}")
        return []

    urls = []
    ext = path.suffix.lower()

    try:
        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            ws = wb.active
            for row in ws.iter_rows(min_col=2, max_col=2, values_only=True):
                val = row[0]
                if val is None:
                    continue
                val = str(val).strip()
                if not val or not val.startswith("http"):
                    continue
                urls.append(val)
            wb.close()
        elif ext == ".xls":
            import xlrd
            wb = xlrd.open_workbook(str(path))
            ws = wb.sheet_by_index(0)
            for row_idx in range(ws.nrows):
                val = ws.cell_value(row_idx, 1)
                if val is None:
                    continue
                val = str(val).strip()
                if not val or not val.startswith("http"):
                    continue
                urls.append(val)
        else:
            print(f"[ERROR] 不支持的 Excel 格式: {ext}")
            return []
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "xlrd 或 openpyxl"
        print(f"[ERROR] 缺少 {missing} 库，请执行: pip install {missing}")
        return []
    except Exception as e:
        print(f"[ERROR] 读取 Excel 失败: {e}")
        return []

    return urls


# ════════════════════════════════════════════════════════
#  公共：单视频处理
# ════════════════════════════════════════════════════════

async def process_single_video(crawler, pipeline, video_url: str,
                                temp_dir: Path, review_dir: Path,
                                num_frames: int, interval: float):
    """抓帧 → 三级漏斗 → 保存证据，单个视频的完整流程。"""
    vid = extract_video_id(video_url)
    print(f"\n  [VIDEO] {vid}")

    frame_dir = temp_dir / "frames" / vid

    # capture_video_frames 内部会做 7s URL 校验，
    # 不匹配时会自动清理 frame_dir 并返回空列表
    frames = await crawler.capture_video_frames(
        video_url, frame_dir,
        num_frames=num_frames, interval=interval,
        random_start=True, skip_verify=True,
    )
    if not frames:
        # capture_video_frames 已经清理了 frame_dir，这里不用再删
        print(f"    [SKIP] 无帧（视频可能不存在或页面跳转）")
        return None

    # ── 三级漏斗 ──
    result = pipeline.process_video(frame_dir, vid, "直链")

    print(f"    [FILTER] {result.total_frames}帧 → "
          f"垃圾过滤:{result.junk_frames} → "
          f"去重:{result.deduped_frames} → "
          f"CLIP召回:{result.clip_candidates} → "
          f"验证通过:{result.verified_matches}")

    if result.candidates:
        max_score = max(c["final_score"] for c in result.candidates)
        print(f"    [HIT] {len(result.candidates)} 个候选! 最高分: {max_score:.0f}")

        out_dir = pipeline.save_evidence(result, review_dir)
        if out_dir:
            print(f"    [SAVED] {out_dir}")
            return out_dir
    else:
        print(f"    [OK] 无候选帧")

    return None


# ════════════════════════════════════════════════════════
#  Phase 1 — 直链模式
# ════════════════════════════════════════════════════════

async def run_phase1_urls(cfg, urls: list[str], headless=False):
    """直链模式：直接传入抖音视频链接，跳过搜索步骤。"""
    temp_dir = BASE_DIR / cfg["output"]["temp_dir"]
    review_dir = BASE_DIR / cfg["output"].get("review_dir", "./review")
    for d in [temp_dir, review_dir]:
        if not d.is_absolute():
            d = BASE_DIR / d
        d.mkdir(parents=True, exist_ok=True)

    pipeline = Phase1Pipeline(cfg)
    n = pipeline.load_library()
    print(f"[INIT] Phase1 三级漏斗管线就绪 (图库: {n} 张)")

    capture_cfg = cfg.get("capture", {})
    num_frames = capture_cfg.get("num_frames", 40)
    interval = capture_cfg.get("interval_seconds", 0.5)

    crawler = DouyinCrawler(headless=headless)
    await crawler.start(use_persistent_context=False)

    review_dirs = []
    try:
        print(f"\n{'='*50}")
        print(f"[URL MODE] 共 {len(urls)} 个视频链接")

        for i, video_url in enumerate(urls):
            video_url = video_url.strip()
            if not video_url:
                continue
            print(f"\n  [{i+1}/{len(urls)}] {video_url}")

            try:
                out_dir = await process_single_video(
                    crawler, pipeline, video_url,
                    temp_dir, review_dir, num_frames, interval,
                )
                if out_dir:
                    review_dirs.append(out_dir)
            except Exception as e:
                print(f"    [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

        # 汇总
        total_candidates = sum(
            len(json.loads((d / "summary.json").read_text(encoding="utf-8")).get("candidates", []))
            for d in review_dirs if (d / "summary.json").exists()
        )
        print(f"\n{'='*50}")
        print(f"[PHASE 1 SUMMARY]")
        print(f"  扫描视频: {len(urls)}")
        print(f"  总候选:   {total_candidates}")
        print(f"  证据目录: {review_dir}")

    finally:
        with suppress_stderr():
            await crawler.close()

    return review_dirs


# ════════════════════════════════════════════════════════
#  Phase 2: AI 二次细筛（可选）
# ════════════════════════════════════════════════════════

async def run_ai_screening(cfg, review_dir_override=None, model_override=None):
    """[可选] 对初筛产出的 review 目录执行 AI 二次细筛。"""
    from src.ai_screener import AIScreener

    ai_cfg = cfg.get("ai_screening", {})
    if not ai_cfg.get("enabled", False):
        print("[AI] AI 二次细筛未启用 (config.yaml → ai_screening.enabled)")
        return []

    screener = AIScreener(
        api_key=ai_cfg.get("api_key", "") or None,
        base_url=ai_cfg.get("base_url", "") or None,
        model=model_override or ai_cfg.get("model", "gpt-4o-mini"),
        min_confidence=ai_cfg.get("min_confidence", 0.3),
        batch_size=ai_cfg.get("batch_size", 5),
    )

    if review_dir_override:
        target = Path(review_dir_override)
        if not target.is_absolute():
            target = BASE_DIR / target
        review_dirs = [target]
    else:
        review_base = BASE_DIR / cfg["output"].get("review_dir", "./review")
        review_dirs = sorted([
            d for d in review_base.iterdir()
            if d.is_dir() and (d / "pairs.json").exists()
        ])

    if not review_dirs:
        print("[AI] 未找到需要 AI 细筛的 review 目录")
        return []

    print(f"\n{'='*50}")
    print(f"[AI SCREENING] 共 {len(review_dirs)} 个目录待审核 (异步并行)")

    all_results = []
    for i, rd in enumerate(review_dirs):
        print(f"\n  [{i+1}/{len(review_dirs)}] {rd.name}")
        try:
            result = await screener.screen_review_package_async(rd)
            all_results.append(result)
            print(f"    确认侵权: {result.confirmed}  疑似: {result.likely}  "
                  f"不太像: {result.unlikely}  排除: {result.rejected}  "
                  f"耗时: {result.cost_seconds}s")
        except Exception as e:
            print(f"    [ERROR] {e}")

    print(f"\n{'='*50}")
    print(f"[AI SCREENING SUMMARY]")
    total_confirmed = sum(r.confirmed for r in all_results)
    total_likely = sum(r.likely for r in all_results)
    total_unlikely = sum(r.unlikely for r in all_results)
    total_rejected = sum(r.rejected for r in all_results)
    print(f"  审核配对: {sum(r.total_pairs for r in all_results)}")
    print(f"  确认侵权: {total_confirmed}")
    print(f"  疑似侵权: {total_likely}")
    print(f"  不太像:   {total_unlikely}")
    print(f"  排除:     {total_rejected}")

    return all_results


# ════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="抖音版权监控 — Phase 1 三级漏斗初筛（直链模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --url https://www.douyin.com/video/xxx          # 单视频直链
  python main.py --url url1 url2 url3                             # 多视频直链
  python main.py --url-file urls.txt                              # 从文件读链接
  python main.py --excel 链接.xlsx                                 # 从 Excel B 列读链接
  python main.py --headless                                       # 无头模式
  python main.py --ai-only                                        # [可选] AI 细筛
        """,
    )
    # 输入来源
    parser.add_argument("--url", type=str, nargs="+", default=None,
                        help="抖音视频直链（支持多个，空格分隔）")
    parser.add_argument("--url-file", type=str, default=None,
                        help="从文本文件读取视频链接（每行一个）")
    parser.add_argument("--excel", type=str, default=None,
                        help="从 Excel 文件 B 列读取视频链接")

    # 通用
    parser.add_argument("--headless", action="store_true",
                        help="无头模式（不弹出浏览器窗口）")
    parser.add_argument("--ai-only", action="store_true",
                        help="仅执行 AI 二次细筛（需已有 review 数据）")
    parser.add_argument("--ai-model", type=str, default=None,
                        help="AI 模型名称（仅 --ai-only 时生效）")
    parser.add_argument("--review-dir", type=str, default=None,
                        help="指定 review 目录（--ai-only 模式下）")
    parser.add_argument("--config", type=str, default=None,
                        help="指定配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── 模式选择 ──
    if args.ai_only:
        asyncio.run(run_ai_screening(cfg, args.review_dir, args.ai_model))
        return

    # 收集 URL 列表
    urls = []
    if args.url:
        urls.extend(args.url)
    if args.url_file:
        url_file = Path(args.url_file)
        if url_file.exists():
            urls.extend(url_file.read_text(encoding="utf-8").strip().splitlines())
            print(f"[URL FILE] 从 {args.url_file} 读取了 {len(urls)} 个链接")
        else:
            print(f"[ERROR] 文件不存在: {args.url_file}")
            sys.exit(1)
    if args.excel:
        excel_urls = read_urls_from_excel(args.excel)
        if excel_urls:
            urls.extend(excel_urls)
            print(f"[EXCEL] 从 {args.excel} 读取了 {len(excel_urls)} 个链接")

    if not urls:
        parser.print_help()
        print("\n[ERROR] 请提供 --url / --url-file / --excel")
        sys.exit(1)

    # 直链模式
    asyncio.run(run_phase1_urls(cfg, urls, args.headless))


if __name__ == "__main__":
    main()
