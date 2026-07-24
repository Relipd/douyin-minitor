"""生成 review 目录汇总仪表盘 HTML。"""
import json
from pathlib import Path

REVIEW_DIR = Path(__file__).parent / "review"
OUTPUT = REVIEW_DIR / "dashboard.html"

entries = []

for sub in sorted(REVIEW_DIR.iterdir()):
    if not sub.is_dir() or sub.name == "dashboard.html":
        continue
    summary_file = sub / "summary.json"
    if not summary_file.exists():
        continue

    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception:
        continue

    candidates = data.get("candidates", [])
    if not candidates:
        continue

    max_score = max(c["final_score"] for c in candidates)
    avg_score = sum(c["final_score"] for c in candidates) / len(candidates)

    best = candidates[0]
    entries.append({
        "video_id": data["video_id"],
        "keyword": data.get("keyword", ""),
        "total_frames": data.get("total_frames", 0),
        "junk_frames": data.get("junk_frames", 0),
        "deduped_frames": data.get("deduped_frames", 0),
        "clip_candidates": data.get("clip_candidates", 0),
        "verified_matches": data.get("verified_matches", 0),
        "candidates": len(candidates),
        "max_score": round(max_score, 1),
        "avg_score": round(avg_score, 1),
        "best_company": best.get("company_name", ""),
        "best_clip": best.get("clip_similarity", 0),
        "best_orb": best.get("orb_good_matches", 0),
        "best_score": best.get("final_score", 0),
    })

# 按综合分排序
entries.sort(key=lambda e: e["max_score"], reverse=True)

# 分类统计
tier_a = [e for e in entries if e["max_score"] >= 80]   # 高度疑似
tier_b = [e for e in entries if 50 <= e["max_score"] < 80]  # 需确认
tier_c = [e for e in entries if e["max_score"] < 50]    # 低可能

# 生成表格行
def make_rows(items):
    rows = ""
    for e in items:
        star = "★★★" if e["max_score"] >= 80 else ("★★" if e["max_score"] >= 50 else "★")
        rows += f"""
        <tr>
            <td><a href="{e['video_id']}/index.html" target="_blank">{e['video_id']}</a></td>
            <td class="score">{e['max_score']:.0f}</td>
            <td>{star}</td>
            <td>{e['total_frames']}</td>
            <td>{e['deduped_frames']}</td>
            <td>{e['clip_candidates']}</td>
            <td>{e['verified_matches']}</td>
            <td>{e['candidates']}</td>
            <td>{e['best_clip']:.2f}</td>
            <td>{e['best_orb']}</td>
            <td>{e['best_company'][:40]}</td>
        </tr>"""
    return rows

html = f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>Phase 1 汇总仪表盘 — 抖音版权监控</title>
<style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{font-family:'Microsoft YaHei',sans-serif;margin:0;background:#0d1117;color:#c9d1d9}}
    .header{{background:#161b22;padding:20px 40px;border-bottom:1px solid #30363d;
             display:flex;justify-content:space-between;align-items:center}}
    .header h1{{font-size:22px;color:#58a6ff}}
    .header .info{{color:#8b949e;font-size:14px}}
    .kpi{{display:grid;grid-template-columns:repeat(6,1fr);gap:15px;padding:25px 40px}}
    .kpi-item{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px;text-align:center}}
    .kpi-item .num{{font-size:28px;font-weight:bold}}
    .kpi-item .label{{font-size:12px;color:#8b949e;margin-top:4px}}
    .kpi-a .num{{color:#ff5252}}
    .kpi-b .num{{color:#ffab40}}
    .kpi-c .num{{color:#7c4dff}}
    .kpi-total .num{{color:#58a6ff}}
    .content{{padding:0 40px 40px}}
    .section-title{{font-size:16px;color:#58a6ff;margin:20px 0 10px 0;padding-bottom:8px;border-bottom:1px solid #21262d}}
    .section-title span{{background:#1f6feb22;padding:2px 10px;border-radius:12px;font-size:13px;margin-left:8px}}
    table{{width:100%;border-collapse:collapse;margin-bottom:30px}}
    th{{background:#21262d;padding:10px 8px;text-align:center;font-size:12px;color:#8b949e;
        position:sticky;top:0;z-index:2}}
    td{{padding:8px;text-align:center;border-bottom:1px solid #21262d;font-size:13px}}
    tr:hover{{background:#161b22}}
    .score{{font-weight:bold;font-size:16px}}
    a{{color:#58a6ff;text-decoration:none;font-family:monospace;font-size:12px}}
    a:hover{{text-decoration:underline}}
    .footer{{text-align:center;padding:20px;color:#484f58;font-size:12px;border-top:1px solid #30363d}}
</style></head><body>

<div class="header">
    <div>
        <h1>🗂 Phase 1 版权监控 — 汇总仪表盘</h1>
    </div>
    <div class="info">总视频: {len(entries)} 个 | 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>

<div class="kpi">
    <div class="kpi-item kpi-total"><div class="num">{len(entries)}</div><div class="label">总候选视频</div></div>
    <div class="kpi-item kpi-a"><div class="num">{len(tier_a)}</div><div class="label">高风险 ≥80分</div></div>
    <div class="kpi-item kpi-b"><div class="num">{len(tier_b)}</div><div class="label">待确认 50-79分</div></div>
    <div class="kpi-item kpi-c"><div class="num">{len(tier_c)}</div><div class="label">低风险 &lt;50分</div></div>
    <div class="kpi-item"><div class="num" style="color:#ffab40">{'%.1f' % (sum(e['max_score'] for e in entries) / len(entries)) if entries else 0}</div><div class="label">平均最高分</div></div>
    <div class="kpi-item"><div class="num" style="color:#3fb950">{sum(e['verified_matches'] for e in entries)}</div><div class="label">总验证通过帧</div></div>
</div>

<div class="content">

<div class="section-title">🔴 高风险 — 评分 ≥ 80（共 {len(tier_a)} 个）<span>需优先人工审核</span></div>
<table>
<thead><tr>
    <th>视频 ID</th><th>评分</th><th>等级</th><th>总帧</th><th>去重后</th><th>CLIP召回</th><th>验证通过</th><th>候选数</th><th>CLIP分数</th><th>ORB匹配</th><th>匹配公司图</th>
</tr></thead><tbody>{make_rows(tier_a)}</tbody></table>

<div class="section-title">🟡 待确认 — 评分 50~79（共 {len(tier_b)} 个）<span>建议人工抽查</span></div>
<table>
<thead><tr>
    <th>视频 ID</th><th>评分</th><th>等级</th><th>总帧</th><th>去重后</th><th>CLIP召回</th><th>验证通过</th><th>候选数</th><th>CLIP分数</th><th>ORB匹配</th><th>匹配公司图</th>
</tr></thead><tbody>{make_rows(tier_b)}</tbody></table>

<div class="section-title">🔵 低可能 — 评分 &lt;50（共 {len(tier_c)} 个）<span>可快速浏览</span></div>
<table>
<thead><tr>
    <th>视频 ID</th><th>评分</th><th>等级</th><th>总帧</th><th>去重后</th><th>CLIP召回</th><th>验证通过</th><th>候选数</th><th>CLIP分数</th><th>ORB匹配</th><th>匹配公司图</th>
</tr></thead><tbody>{make_rows(tier_c)}</tbody></table>

</div>

<div class="footer">Phase 1 三级漏斗：垃圾帧过滤 → SSIM 去重 → CLIP 语义召回 → ORB 特征验证 | 评分 = CLIP×40% + ORB×60%</div>

</body></html>"""

OUTPUT.write_text(html, encoding="utf-8")
print(f"Dashboard saved: {OUTPUT}")
print(f"Total: {len(entries)} | High-risk: {len(tier_a)} | Pending: {len(tier_b)} | Low: {len(tier_c)}")
print(f"Avg max score: {sum(e['max_score'] for e in entries) / len(entries):.1f}" if entries else "")
