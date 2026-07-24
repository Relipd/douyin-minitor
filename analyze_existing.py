"""离线分析脚本 — 对已截取的帧执行三级漏斗。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.phase1 import Phase1Pipeline
from src.utils import load_config

def main():
    cfg = load_config(Path("./config.yaml"))
    pipeline = Phase1Pipeline(cfg)
    n = pipeline.load_library()
    print(f"[INIT] 图库加载完成: {n} 张")

    frame_base = Path("./temp/frames")
    review_dir = Path("./review")
    review_dir.mkdir(exist_ok=True)

    for video_dir in sorted(frame_base.iterdir()):
        if not video_dir.is_dir():
            continue
        vid = video_dir.name
        print(f"\n{'='*50}")
        print(f"[VIDEO] {vid}")
        result = pipeline.process_video(video_dir, vid, "除螨喷雾")
        print(f"  帧数: {result.total_frames} -> 垃圾:{result.junk_frames} -> 去重:{result.deduped_frames} -> CLIP:{result.clip_candidates} -> 验证:{result.verified_matches}")
        if result.candidates:
            out = pipeline.save_evidence(result, review_dir)
            print(f"  [SAVED] {out} ({len(result.candidates)} 个候选)")
            for c in result.candidates:
                print(f"    {c['company_name']}  CLIP={c['clip_similarity']:.3f}  ORB={c['orb_good_matches']}pts  score={c['final_score']:.0f}")
        else:
            print(f"  [OK] 无候选")

if __name__ == "__main__":
    main()
