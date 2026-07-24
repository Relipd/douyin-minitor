"""独立 AI 二次细筛脚本。

对已有的 review 目录执行：
  1. 生成配对对比图（如果 pairs.json 不存在）
  2. AI 多模态审核
  3. 输出结果

用法:
    python run_ai_screening.py                          # 审核 review/ 下所有目录
    python run_ai_screening.py --review-dir ./review/video_123  # 审核指定目录
    python run_ai_screening.py --api-key sk-xxx --model gpt-4o  # 指定 API
    python run_ai_screening.py --base-url https://api.deepseek.com/v1 --model deepseek-chat  # DeepSeek
"""
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

from src.pair_generator import PairGenerator
from src.ai_screener import AIScreener
from src.utils import load_config as _load_config, resolve_path as _resolve_path

BASE_DIR = Path(__file__).parent

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


def main():
    parser = argparse.ArgumentParser(
        description="AI 二次细筛独立脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_ai_screening.py
  python run_ai_screening.py --review-dir ./review/video_123
  python run_ai_screening.py --api-key sk-xxx --model gpt-4o
  python run_ai_screening.py --base-url https://api.deepseek.com/v1 --model deepseek-chat
        """,
    )
    parser.add_argument("--review-dir", type=str, default=None,
                        help="指定 review 目录（默认处理 review/ 下所有子目录）")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API 密钥（覆盖 config）")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API 地址（覆盖 config）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称（覆盖 config）")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="最低置信度阈值")
    parser.add_argument("--skip-pairs", action="store_true",
                        help="跳过配对图生成（已有 pairs.json 时）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅生成配对图，不执行 AI 审核")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ai_cfg = cfg.get("ai_screening", {})

    # ── 确定 review 目录 ──
    if args.review_dir:
        review_dir = Path(args.review_dir)
        if not review_dir.is_absolute():
            review_dir = BASE_DIR / review_dir
        review_dirs = [review_dir]
    else:
        review_base = Path(cfg.get("output", {}).get("review_dir", "./review"))
        if not review_base.is_absolute():
            review_base = BASE_DIR / review_base
        if not review_base.exists():
            print(f"[ERROR] Review 目录不存在: {review_base}")
            return
        review_dirs = sorted([
            d for d in review_base.iterdir()
            if d.is_dir()
        ])

    if not review_dirs:
        print("[ERROR] 未找到 review 目录")
        return

    print(f"[INFO] 发现 {len(review_dirs)} 个 review 目录")

    # ── Phase 1: 生成配对图 ──
    if not args.skip_pairs:
        full_dir = resolve_path(cfg, "company_images_path", "./company_images")
        crop_dir = resolve_path(cfg, "crop_images_path", "./crop_images")

        print(f"\n{'='*50}")
        print(f"[PHASE 1] 生成配对对比图")
        print(f"  公司图目录: {full_dir}")
        print(f"  裁剪图目录: {crop_dir}")

        generator = PairGenerator(
            company_images_dir=full_dir,
            crop_images_dir=crop_dir,
        )

        for rd in review_dirs:
            pairs_json = rd / "pairs.json"
            if pairs_json.exists() and not args.dry_run:
                print(f"  [SKIP] {rd.name} (pairs.json 已存在)")
                continue
            try:
                generator.generate_from_suspect_detector(rd)
                # 重新读取统计
                with open(pairs_json, encoding="utf-8") as f:
                    meta = json.load(f)
                print(f"  [OK] {rd.name}: {meta['total_pairs']} 个配对")
            except FileNotFoundError:
                print(f"  [SKIP] {rd.name} (无 summary.json)")
            except Exception as e:
                print(f"  [WARN] {rd.name}: {e}")

    if args.dry_run:
        print("\n[DRY RUN] 配对图生成完毕，跳过 AI 审核")
        return

    # ── Phase 2: AI 审核 ──
    print(f"\n{'='*50}")
    print(f"[PHASE 2] AI 二次细筛")

    screener = AIScreener(
        api_key=args.api_key or ai_cfg.get("api_key", "") or None,
        base_url=args.base_url or ai_cfg.get("base_url", "") or None,
        model=args.model or ai_cfg.get("model", "gpt-4o-mini"),
        min_confidence=args.min_confidence or ai_cfg.get("min_confidence", 0.3),
        batch_size=ai_cfg.get("batch_size", 5),
    )

    # 检查 API key
    if not screener.api_key:
        print("[ERROR] 未提供 API 密钥。请通过以下方式之一提供：")
        print("  1. --api-key sk-xxx")
        print("  2. 环境变量 OPENAI_API_KEY")
        print("  3. config.yaml → ai_screening.api_key")
        return

    print(f"  模型: {screener.model}")
    print(f"  API: {screener.base_url}")

    # 过滤有 pairs.json 的目录
    valid_dirs = [d for d in review_dirs if (d / "pairs.json").exists()]
    if not valid_dirs:
        print("[ERROR] 未找到包含 pairs.json 的 review 目录")
        return

    print(f"  待审核: {len(valid_dirs)} 个目录\n")

    all_results = []
    for i, rd in enumerate(valid_dirs):
        print(f"  [{i+1}/{len(valid_dirs)}] {rd.name}")
        try:
            result = screener.screen_review_package(rd)
            all_results.append(result)
            print(f"    OK 确认: {result.confirmed}  疑似: {result.likely}  "
                  f"不像: {result.unlikely}  排除: {result.rejected}  "
                  f"({result.cost_seconds}s)")
        except Exception as e:
            print(f"    ERR: {e}")

    # ── 最终汇总 ──
    print(f"\n{'='*50}")
    print(f"[FINAL SUMMARY]")
    total_pairs = sum(r.total_pairs for r in all_results)
    total_confirmed = sum(r.confirmed for r in all_results)
    total_likely = sum(r.likely for r in all_results)
    total_unlikely = sum(r.unlikely for r in all_results)
    total_rejected = sum(r.rejected for r in all_results)
    print(f"  审核配对总数: {total_pairs}")
    print(f"  确认侵权:     {total_confirmed}")
    print(f"  疑似侵权:     {total_likely}")
    print(f"  不太像:       {total_unlikely}")
    print(f"  排除:         {total_rejected}")
    print(f"\n  详细结果请查看各 review 目录下的 ai_verdicts.html")


if __name__ == "__main__":
    main()
