"""端到端集成测试。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.pair_generator import PairGenerator
from src.ai_screener import AIScreener

BASE = Path(__file__).parent

def main():
    # === 1. 配对图生成 ===
    print("=" * 50)
    print("[TEST 1] PairGenerator 配对图生成")
    gen = PairGenerator(
        company_images_dir=str(BASE / "company_images"),
        crop_images_dir=str(BASE / "crop_images"),
    )
    print(f"  公司图索引: {len(gen._image_index)} 张")
    for name, path in list(gen._image_index.items())[:5]:
        print(f"    {name} -> {path}")
    assert len(gen._image_index) > 0, "公司图索引为空"

    # === 2. AI 审核器初始化 ===
    print()
    print("=" * 50)
    print("[TEST 2] AIScreener 初始化")
    screener = AIScreener(api_key="test-key", model="gpt-4o-mini")
    print(f"  模型: {screener.model}")
    print(f"  API: {screener.base_url}")
    print(f"  最低置信度: {screener.min_confidence}")

    # === 3. Prompt 构建 ===
    print()
    print("=" * 50)
    print("[TEST 3] Prompt 构建")
    meta = {"phash_distance": 12, "good_matches": 30, "inliers": 15}
    prompt = screener._build_prompt(meta)
    print(f"  Prompt 长度: {len(prompt)} 字符")
    assert "pHash" in prompt
    assert "12" in prompt
    assert "30" in prompt
    assert "15" in prompt
    print("  OK: Prompt 包含所有背景信息")

    # === 4. 响应解析 ===
    print()
    print("=" * 50)
    print("[TEST 4] 响应解析")
    tests = [
        (
            '{"is_match": true, "confidence": 0.95, "reason": "product matches"}',
            True, 0.95,
        ),
        (
            '{"is_match": false, "confidence": 0.1, "reason": "different product"}',
            False, 0.1,
        ),
    ]
    for resp, exp_match, exp_conf in tests:
        is_match, conf, _ = screener._parse_response(resp)
        assert is_match == exp_match, f"mismatch: {is_match} != {exp_match}"
        assert abs(conf - exp_conf) < 0.05, f"conf mismatch: {conf} != {exp_conf}"
        print(f"  OK: conf={conf}, match={is_match}")

    # markdown code block
    resp_codeblock = '```json\n{"is_match": true, "confidence": 0.7, "reason": "suspect"}\n```'
    is_match, conf, _ = screener._parse_response(resp_codeblock)
    assert is_match is True
    assert abs(conf - 0.7) < 0.05
    print("  OK: markdown code block 解析正确")

    # empty
    is_match, conf, reason = screener._parse_response("")
    assert is_match is False
    assert conf == 0.0
    print("  OK: 空响应处理正确")

    # === 5. 分类逻辑 ===
    print()
    print("=" * 50)
    print("[TEST 5] 分类逻辑")
    cases = [
        (0.95, True, "confirmed"),
        (0.7, True, "likely"),
        (0.4, True, "unlikely"),
        (0.1, False, "rejected"),
    ]
    for conf, match, expected in cases:
        cat = screener._categorize(conf, match)
        assert cat == expected, f"{cat} != {expected}"
        print(f"  OK: conf={conf}, match={match} -> {cat}")

    # === 6. 数据结构 ===
    print()
    print("=" * 50)
    print("[TEST 6] 数据结构")
    from src.ai_screener import AIVerdict, AIScreeningResult
    v = AIVerdict(
        pair_index=1, pair_image="p.jpg", frame_file="f.jpg",
        company_file="c.jpg", is_match=True, confidence=0.9,
        reason="test", category="confirmed",
    )
    assert v.is_match is True
    print(f"  OK: AIVerdict 创建成功, category={v.category}")

    r = AIScreeningResult(video_id="v1", keyword="kw", total_pairs=3)
    r.confirmed = 2
    r.rejected = 1
    assert r.total_pairs == 3
    assert r.confirmed + r.rejected == 3
    print(f"  OK: AIScreeningResult 创建成功, total={r.total_pairs}")

    # === 7. 公司图查找 ===
    print()
    print("=" * 50)
    print("[TEST 7] 公司图查找")
    for img_name in ["11(1).jpeg", "22(1).jpg", "33.jpg", "44.jpeg"]:
        found = gen.find_company_image(img_name)
        assert found is not None, f"找不到: {img_name}"
        print(f"  OK: {img_name} -> {found.name}")

    # crop images
    for img_name in ["33-removebg-preview-removebg-preview.png"]:
        found = gen.find_company_image(img_name)
        assert found is not None, f"找不到 crop: {img_name}"
        print(f"  OK: {img_name} -> {found.name}")

    print()
    print("=" * 50)
    print("[RESULT] ALL 7 TESTS PASSED!")

if __name__ == "__main__":
    main()
