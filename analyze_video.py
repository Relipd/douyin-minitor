"""单视频 Phase 1 三级漏斗初筛工具。

用法:
    python analyze_video.py <视频URL>
    python analyze_video.py <视频URL> --frames 60 --interval 0.3
    python analyze_video.py <视频URL> --cleanup       # 分析后清理临时帧
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.phase1 import Phase1Pipeline
from src.crawler import DouyinCrawler
from src.utils import load_config as _load_config, resolve_path as _resolve_path

BASE_DIR = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_video")


def load_config(path=None):
    return _load_config(path, base_dir=BASE_DIR)


def resolve_path(cfg, key, default=None):
    return _resolve_path(cfg, key, base_dir=BASE_DIR, default=default)


def extract_video_id(url: str) -> str:
    """从抖音 URL 中提取视频 ID。"""
    if "/video/" in url:
        return url.split("/video/")[-1].split("?")[0].rstrip("/")
    return url.rstrip("/").split("/")[-1].split("?")[0]


@contextmanager
def suppress_stderr():
    """静默 stderr（用于屏蔽 Playwright/Chromium 的噪音日志）。"""
    import os
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        devnull.close()
        sys.stderr = old_stderr


async def analyze_single_video(
    video_url: str,
    num_frames: int,
    interval: float,
    headless: bool,
    cfg: dict,
    cleanup: bool = False,
) -> dict | None:
    """用 Phase 1 三级漏斗分析单个抖音视频。

    流程:
        1. 加载公司图库 (CLIP embedding + ORB 特征)
        2. 抓取视频帧
        3. 三级漏斗初筛 (垃圾帧过滤 → SSIM 去重 → CLIP+ORB)
        4. 保存证据包 + 生成 HTML 报告
    """
    video_id = extract_video_id(video_url)

    # 准备目录
    temp_dir = BASE_DIR / cfg["output"]["temp_dir"]
    review_dir = BASE_DIR / cfg["output"].get("review_dir", "./review")
    for d in [temp_dir, review_dir]:
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── 1. 加载图库 ──
    pipeline = Phase1Pipeline(cfg)
    n = pipeline.load_library()
    logger.info(f"Phase 1 三级漏斗就绪 (图库: {n} 张)")

    # ── 2. 抓帧 ──
    logger.info(f"视频: {video_url}")
    logger.info(f"抓帧参数: {num_frames} 帧, 间隔 {interval}s")

    crawler = DouyinCrawler(headless=headless, user_data_dir=str(BASE_DIR / "browser_data"))
    await crawler.start()

    frame_dir = temp_dir / "frames" / video_id
    try:
        with suppress_stderr():
            frames = await crawler.capture_video_frames(
                video_url, frame_dir,
                num_frames=num_frames, interval=interval,
                random_start=False,
            )
        if not frames:
            logger.error("未能抓取到任何帧")
            return None

        logger.info(f"成功抓取 {len(frames)} 帧")

        # ── 3. 三级漏斗初筛 ──
        t_screen = time.time()
        result = pipeline.process_video(frame_dir, video_id, keyword="single")
        screen_time = time.time() - t_screen

        print(f"    [FILTER] {result.total_frames}帧 → "
              f"垃圾过滤:{result.junk_frames} → "
              f"去重:{result.deduped_frames} → "
              f"CLIP召回:{result.clip_candidates} → "
              f"验证通过:{result.verified_matches}")

        if not result.candidates:
            logger.info(f"未发现疑似帧（共检查 {result.total_frames} 帧, 耗时 {screen_time:.1f}s）")
            return {
                "video_id": video_id,
                "status": "clean",
                "frames_checked": result.total_frames,
                "suspects": 0,
                "time_screen": round(screen_time, 2),
                "time_total": round(time.time() - t_start, 2),
            }

        max_score = max(c["final_score"] for c in result.candidates)
        logger.info(f"初筛发现 {len(result.candidates)} 个候选 (最高分 {max_score:.0f}, 耗时 {screen_time:.1f}s)")

        # 保存初筛证据
        out_dir = pipeline.save_evidence(result, review_dir)
        logger.info(f"初筛证据已保存: {out_dir}")

        # ── 4. 最终汇总 ──
        elapsed = time.time() - t_start
        result_dict = {
            "video_id": video_id,
            "status": "suspect",
            "frames_checked": result.total_frames,
            "suspects": len(result.candidates),
            "max_score": round(max_score, 1),
            "time_screen": round(screen_time, 2),
            "time_total": round(elapsed, 2),
            "evidence_dir": str(out_dir),
        }

        # 保存分析元数据
        meta_path = out_dir / "analysis_meta.json"
        meta_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        _print_summary(result_dict)
        return result_dict

    finally:
        with suppress_stderr():
            await crawler.close()

        # 清理临时帧
        if cleanup and temp_dir.exists():
            import shutil
            frames_dir = temp_dir / "frames" / video_id
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)
                logger.info(f"已清理临时帧: {frames_dir}")


def _print_summary(result: dict):
    """打印最终分析摘要。"""
    status = result["status"]
    if status == "clean":
        logger.info("=" * 50)
        logger.info("✅ 分析完成 — 未发现疑似内容")
        logger.info(f"   检查帧数: {result['frames_checked']}")
        logger.info(f"   总耗时: {result['time_total']}s")
        logger.info("=" * 50)
    else:
        logger.info("=" * 50)
        logger.info(f"⚠️  分析完成 — 发现 {result['suspects']} 个疑似帧")
        logger.info(f"   最高疑似分: {result['max_score']}")
        logger.info(f"   检查帧数: {result['frames_checked']}")
        logger.info(f"   证据目录: {result['evidence_dir']}")
        logger.info(f"   总耗时: {result['time_total']}s")
        logger.info("=" * 50)


# ── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="抖音视频 Phase 1 三级漏斗初筛工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze_video.py https://www.douyin.com/video/123456
  python analyze_video.py https://www.douyin.com/video/123456 --frames 60 --interval 0.3
  python analyze_video.py https://www.douyin.com/video/123456 --cleanup
        """,
    )
    parser.add_argument("url", help="抖音视频 URL")
    parser.add_argument("--frames", type=int, default=40, help="抓取帧数 (默认: 40)")
    parser.add_argument("--interval", type=float, default=0.5, help="帧间隔秒数 (默认: 0.5)")
    parser.add_argument("--headless", action="store_true", help="无头模式运行浏览器")
    parser.add_argument("--cleanup", action="store_true", help="分析完成后清理临时帧文件")
    parser.add_argument("--config", type=str, default=None, help="自定义配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = asyncio.run(analyze_single_video(
        video_url=args.url,
        num_frames=args.frames,
        interval=args.interval,
        headless=args.headless,
        cfg=cfg,
        cleanup=args.cleanup,
    ))

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
