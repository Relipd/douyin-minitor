"""AI 多模态二次细筛模块。

流程：
  1. 读取 Python 初筛产出的配对对比图（帧 vs 公司图）
  2. 将图片发送给多模态 AI（支持 OpenAI / 兼容 API）
  3. AI 返回结构化判定：是否侵权、置信度、理由
  4. 汇总结果供人工最终审核
"""
import asyncio
import base64
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AIVerdict:
    """AI 对单个配对的判定结果。"""
    pair_index: int
    pair_image: str
    frame_file: str
    company_file: str
    is_match: bool            # AI 判定是否为同一内容
    confidence: float         # 0~1 置信度
    reason: str               # AI 给出的理由
    category: str             # "confirmed" / "likely" / "unlikely" / "rejected"
    phash_distance: int = 0
    good_matches: int = 0
    inliers: int = 0
    # Python 初筛分数
    screening_score: float = 0.0


@dataclass
class AIScreeningResult:
    """一次 AI 细筛的完整结果。"""
    video_id: str
    keyword: str
    total_pairs: int
    confirmed: int = 0
    likely: int = 0
    unlikely: int = 0
    rejected: int = 0
    verdicts: list = field(default_factory=list)
    ai_model: str = ""
    cost_seconds: float = 0.0


class AIScreener:
    """多模态 AI 二次细筛器。

    支持的 API：
      - OpenAI GPT-4o / GPT-4o-mini（默认）
      - 任何兼容 OpenAI 格式的 API（如 DeepSeek、本地部署）
    """

    CATEGORIES = {
        "confirmed": (0.8, 1.0),   # 高度确认侵权
        "likely":    (0.5, 0.8),   # 疑似侵权
        "unlikely":  (0.2, 0.5),   # 不太像
        "rejected":  (0.0, 0.2),   # 明确否定
    }

    def __init__(self,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = "gpt-4o-mini",
                 max_retries: int = 3,
                 retry_delay: float = 5.0,
                 timeout: int = 30,
                 min_confidence: float = 0.3,
                 batch_size: int = 5):
        """
        Args:
            api_key: API 密钥。未提供时从环境变量 OPENAI_API_KEY 读取。
            base_url: API 地址。支持 OpenAI / DeepSeek / 本地部署等。
            model: 模型名称。
            max_retries: 最大重试次数。
            retry_delay: 重试间隔秒数。
            timeout: 单次请求超时秒数。
            min_confidence: 最低置信度阈值（低于此值的判定标记为 "uncertain"）。
            batch_size: 每批发送的配对数（控制并发和 token 消耗）。
        """
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.min_confidence = min_confidence
        self.batch_size = batch_size

    # ── 主入口 ──────────────────────────────────────────

    async def screen_review_package_async(self, review_dir: Path) -> AIScreeningResult:
        """异步版: 对一个 review 目录下的所有配对进行 AI 二次细筛。

        使用 httpx 并行处理，比同步版快 5-10 倍。
        """
        import httpx

        start_time = time.time()

        pairs_json = review_dir / "pairs.json"
        if not pairs_json.exists():
            raise FileNotFoundError(f"pairs.json not found in {review_dir}")

        with open(pairs_json, encoding="utf-8") as f:
            meta = json.load(f)

        video_id = meta.get("video_id", "")
        keyword = meta.get("keyword", "")
        pairs = meta.get("pairs", [])

        result = AIScreeningResult(
            video_id=video_id,
            keyword=keyword,
            total_pairs=len(pairs),
            ai_model=self.model,
        )

        if not pairs:
            return result

        # 过滤有效配对
        valid_pairs = []
        for pair in pairs:
            pair_img_path = review_dir / pair["pair_image"]
            if pair_img_path.exists():
                valid_pairs.append((pair_img_path, pair))

        if not valid_pairs:
            result.cost_seconds = round(time.time() - start_time, 2)
            return result

        # 并行 AI 审核
        sem = asyncio.Semaphore(self.batch_size)

        async def _judge_one(client, pair_img_path, pair_meta):
            async with sem:
                return await self._judge_pair_async(client, pair_img_path, pair_meta)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                _judge_one(client, img_path, pair_meta)
                for img_path, pair_meta in valid_pairs
            ]
            verdicts = await asyncio.gather(*tasks, return_exceptions=True)

        for v in verdicts:
            if isinstance(v, Exception):
                logger.warning(f"AI 审核异常: {v}")
                continue
            if v is None:
                continue
            result.verdicts.append(v)
            if v.category == "confirmed":
                result.confirmed += 1
            elif v.category == "likely":
                result.likely += 1
            elif v.category == "unlikely":
                result.unlikely += 1
            else:
                result.rejected += 1

        result.cost_seconds = round(time.time() - start_time, 2)
        self._save_ai_verdicts(review_dir, result)
        return result

    async def _judge_pair_async(self, client, pair_img_path: Path, pair_meta: dict):
        """异步版: 对单个配对图调用 AI 进行判定。"""
        try:
            img_b64 = self._encode_image(pair_img_path)
            prompt = self._build_prompt(pair_meta)
            response_text = await self._call_vision_api_async(client, prompt, img_b64)
            is_match, confidence, reason = self._parse_response(response_text)
            category = self._categorize(confidence, is_match)
            return AIVerdict(
                pair_index=pair_meta.get("index", 0),
                pair_image=pair_meta.get("pair_image", ""),
                frame_file=pair_meta.get("frame_file", ""),
                company_file=pair_meta.get("company_file", ""),
                is_match=is_match,
                confidence=round(confidence, 3),
                reason=reason,
                category=category,
                phash_distance=pair_meta.get("phash_distance", 0),
                good_matches=pair_meta.get("good_matches", 0),
                inliers=pair_meta.get("inliers", 0),
            )
        except Exception as e:
            logger.warning(f"AI 审核配对 {pair_meta.get('pair_image', '?')} 失败: {e}")
            return None

    async def _call_vision_api_async(self, client, prompt: str, image_b64: str) -> str:
        """异步版: 调用 OpenAI 兼容的多模态 API。"""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(self.max_retries):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 429:
                    wait = self.retry_delay * (attempt + 1) * 2
                    logger.warning(f"API 限流，等待 {wait}s 后重试...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    await asyncio.sleep(self.retry_delay)
                    continue
                resp.raise_for_status()
                body = resp.json()
                return body["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"API 请求异常 (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
        return ""

    def screen_review_package(self, review_dir: Path) -> AIScreeningResult:
        """对一个 review 目录下的所有配对进行 AI 二次细筛。

        Args:
            review_dir: screening.py 产出的 review 目录，内含 pairs.json + 配对图。

        Returns:
            AIScreeningResult 包含所有配对的 AI 判定。
        """
        start_time = time.time()

        pairs_json = review_dir / "pairs.json"
        if not pairs_json.exists():
            raise FileNotFoundError(f"pairs.json not found in {review_dir}")

        with open(pairs_json, encoding="utf-8") as f:
            meta = json.load(f)

        video_id = meta.get("video_id", "")
        keyword = meta.get("keyword", "")
        pairs = meta.get("pairs", [])

        result = AIScreeningResult(
            video_id=video_id,
            keyword=keyword,
            total_pairs=len(pairs),
            ai_model=self.model,
        )

        if not pairs:
            return result

        # 逐对 AI 审核
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            for pair in batch:
                pair_img_path = review_dir / pair["pair_image"]
                if not pair_img_path.exists():
                    logger.warning(f"配对图不存在: {pair_img_path}")
                    continue

                verdict = self._judge_pair(
                    pair_img_path=pair_img_path,
                    pair_meta=pair,
                )
                result.verdicts.append(verdict)

                # 统计
                if verdict.category == "confirmed":
                    result.confirmed += 1
                elif verdict.category == "likely":
                    result.likely += 1
                elif verdict.category == "unlikely":
                    result.unlikely += 1
                else:
                    result.rejected += 1

        result.cost_seconds = round(time.time() - start_time, 2)

        # 保存 AI 审核结果
        self._save_ai_verdicts(review_dir, result)

        return result

    # ── 对比图判定 ──────────────────────────────────────

    def _judge_pair(self, pair_img_path: Path, pair_meta: dict) -> AIVerdict:
        """对单个配对图调用 AI 进行判定。"""
        img_b64 = self._encode_image(pair_img_path)

        prompt = self._build_prompt(pair_meta)

        # 调用 API
        response_text = self._call_vision_api(prompt, img_b64)

        # 解析 AI 返回
        is_match, confidence, reason = self._parse_response(response_text)

        # 分类
        category = self._categorize(confidence, is_match)

        return AIVerdict(
            pair_index=pair_meta.get("index", 0),
            pair_image=pair_meta.get("pair_image", ""),
            frame_file=pair_meta.get("frame_file", ""),
            company_file=pair_meta.get("company_file", ""),
            is_match=is_match,
            confidence=round(confidence, 3),
            reason=reason,
            category=category,
            phash_distance=pair_meta.get("phash_distance", 0),
            good_matches=pair_meta.get("good_matches", 0),
            inliers=pair_meta.get("inliers", 0),
        )

    def _build_prompt(self, pair_meta: dict) -> str:
        """构建发送给 AI 的审核 prompt。"""
        return f"""你是一个版权侵权检测专家。请仔细对比这张图片中的两张图：

**左侧**：从抖音视频中截取的视频帧
**右侧**：公司版权原图

背景信息：
- pHash 汉明距离: {pair_meta.get('phash_distance', 'N/A')}（越小越相似）
- ORB 特征匹配点数: {pair_meta.get('good_matches', 'N/A')}
- RANSAC 内点数: {pair_meta.get('inliers', 'N/A')}

请从以下角度分析：
1. 视频帧中是否出现了与右侧公司图相同或高度相似的产品/品牌内容？
2. 是否存在以下情况之一：
   - 直接使用了公司图片（如产品展示、广告素材）
   - 视频中的产品/品牌与公司图一致
   - 只是巧合的颜色或形状相似（非侵权）
3. 图片质量是否足以做出判断？

请严格按以下 JSON 格式返回结果（不要返回其他内容）：
```json
{{
  "is_match": true/false,
  "confidence": 0.0~1.0,
  "reason": "简要说明判定理由（50字以内）"
}}
```

注意：
- is_match=true 表示你认为视频帧中确实包含了公司版权内容
- confidence 表示你对判定的把握程度（1.0=完全确定）
- 如果图片模糊无法判断，设 is_match=false, confidence=0.1"""

    # ── API 调用 ────────────────────────────────────────

    def _call_vision_api(self, prompt: str, image_b64: str) -> str:
        """调用 OpenAI 兼容的多模态 API。"""
        import urllib.request
        import urllib.error

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = json.dumps(payload).encode("utf-8")

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body_text = ""
                try:
                    body_text = e.read().decode("utf-8")
                except Exception:
                    pass
                logger.warning(f"API 请求失败 (attempt {attempt+1}): HTTP {e.code} {body_text[:200]}")
                if e.code == 429:
                    wait = self.retry_delay * (attempt + 1) * 2
                    logger.warning(f"  限流，等待 {wait}s 后重试...")
                    time.sleep(wait)
                elif e.code >= 500:
                    time.sleep(self.retry_delay)
                else:
                    raise
            except Exception as e:
                logger.warning(f"API 请求异常 (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise

        return ""

    # ── 结果解析 ────────────────────────────────────────

    def _parse_response(self, response_text: str) -> tuple[bool, float, str]:
        """解析 AI 返回的 JSON 结果。"""
        if not response_text:
            return False, 0.0, "AI 无响应"

        # 尝试提取 JSON
        text = response_text.strip()

        # 处理 markdown 代码块
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        # 尝试解析 JSON
        try:
            data = json.loads(text)
            is_match = bool(data.get("is_match", False))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(data.get("reason", ""))[:200]
            return is_match, confidence, reason
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # fallback: 文本解析
        text_lower = response_text.lower()
        if "is_match" in text_lower:
            is_match = "true" in text_lower and "false" not in text_lower
        else:
            is_match = any(kw in text_lower for kw in ["侵权", "相同", "一致", "匹配", "使用了"])
            is_match = is_match and not any(kw in text_lower for kw in ["不侵权", "不相同", "不一致", "不匹配", "没有使用"])

        confidence = 0.5 if is_match else 0.3
        return is_match, confidence, f"文本解析: {response_text[:100]}"

    def _categorize(self, confidence: float, is_match: bool) -> str:
        """根据置信度和判定结果分类。"""
        if not is_match and confidence < 0.3:
            return "rejected"
        if is_match and confidence >= 0.8:
            return "confirmed"
        if is_match and confidence >= 0.5:
            return "likely"
        if is_match and confidence >= 0.3:
            return "unlikely"
        return "rejected"

    # ── 结果保存 ────────────────────────────────────────

    def _save_ai_verdicts(self, review_dir: Path, result: AIScreeningResult):
        """保存 AI 审核结果到 JSON 和 HTML。"""
        # JSON
        ai_json = {
            "video_id": result.video_id,
            "keyword": result.keyword,
            "ai_model": result.ai_model,
            "cost_seconds": result.cost_seconds,
            "summary": {
                "total": result.total_pairs,
                "confirmed": result.confirmed,
                "likely": result.likely,
                "unlikely": result.unlikely,
                "rejected": result.rejected,
            },
            "verdicts": [asdict(v) for v in result.verdicts],
        }
        out_json = review_dir / "ai_verdicts.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ai_json, f, ensure_ascii=False, indent=2)

        # HTML
        self._write_ai_html(review_dir, ai_json)

        logger.info(f"AI 审核结果已保存: {out_json}")

    def _write_ai_html(self, review_dir: Path, data: dict):
        """生成 AI 审核结果 HTML 报告。"""
        summary = data["summary"]
        rows = ""
        for v in data["verdicts"]:
            cat = v["category"]
            color_map = {
                "confirmed": "#d32f2f",
                "likely": "#ff9800",
                "unlikely": "#4caf50",
                "rejected": "#888",
            }
            color = color_map.get(cat, "#888")
            label_map = {
                "confirmed": "确认侵权",
                "likely": "疑似侵权",
                "unlikely": "不太像",
                "rejected": "排除",
            }
            label = label_map.get(cat, cat)
            conf_pct = int(v["confidence"] * 100)

            rows += f"""
            <tr>
                <td>{v['pair_index']}</td>
                <td><img src="{v['pair_image']}" class="thumb" onclick="this.classList.toggle('zoomed')"></td>
                <td style="color:{color};font-weight:bold;font-size:16px">{label}</td>
                <td>{conf_pct}%</td>
                <td style="text-align:left;font-size:13px">{v['reason']}</td>
                <td>{v['phash_distance']}</td>
                <td>{v['good_matches']}</td>
                <td>{v['inliers']}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>AI 细筛结果 - {data['video_id']}</title>
<style>
    body{{font-family:'Microsoft YaHei',sans-serif;margin:20px;background:#111;color:#eee}}
    h2{{color:#4fc3f7}}
    .summary{{background:#1a1a2e;padding:20px;border-radius:10px;margin-bottom:20px;
              display:grid;grid-template-columns:repeat(5,1fr);gap:15px;text-align:center}}
    .stat{{padding:10px;border-radius:8px}}
    .stat .num{{font-size:28px;font-weight:bold}}
    .stat .label{{font-size:12px;color:#aaa;margin-top:4px}}
    .s-confirmed{{background:#3d1f1f}} .s-confirmed .num{{color:#ff5252}}
    .s-likely{{background:#3d2e1f}} .s-likely .num{{color:#ffab40}}
    .s-unlikely{{background:#1f3d1f}} .s-unlikely .num{{color:#69f0ae}}
    .s-rejected{{background:#2a2a2a}} .s-rejected .num{{color:#888}}
    .s-total{{background:#1a2a3d}} .s-total .num{{color:#4fc3f7}}
    table{{border-collapse:collapse;width:100%}}
    th{{background:#333;padding:10px;position:sticky;top:0}}
    td{{padding:8px;text-align:center;border-bottom:1px solid #333}}
    .thumb{{max-height:300px;cursor:pointer;transition:all 0.3s;border-radius:4px}}
    .zoomed{{max-height:90vh;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:999}}
    tr:hover{{background:#222}}
    .info{{color:#888;margin-bottom:15px}}
</style></head><body>
<h2>AI 二次细筛结果</h2>
<div class="info">
    视频: {data['video_id']} | 关键词: {data['keyword']} | 模型: {data['ai_model']} | 耗时: {data['cost_seconds']}s
</div>
<div class="summary">
    <div class="stat s-total"><div class="num">{summary['total']}</div><div class="label">总配对数</div></div>
    <div class="stat s-confirmed"><div class="num">{summary['confirmed']}</div><div class="label">确认侵权</div></div>
    <div class="stat s-likely"><div class="num">{summary['likely']}</div><div class="label">疑似侵权</div></div>
    <div class="stat s-unlikely"><div class="num">{summary['unlikely']}</div><div class="label">不太像</div></div>
    <div class="stat s-rejected"><div class="num">{summary['rejected']}</div><div class="label">排除</div></div>
</div>
<table><thead><tr>
    <th>#</th><th>对比图</th><th>AI判定</th><th>置信度</th><th>理由</th>
    <th>pHash</th><th>ORB匹配</th><th>内点</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""

        (review_dir / "ai_verdicts.html").write_text(html, encoding="utf-8")

    # ── 辅助 ────────────────────────────────────────────

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """将图片编码为 base64 字符串。"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ── 批量处理 ────────────────────────────────────────

    def screen_all_reviews(self, review_base_dir: Path) -> list[AIScreeningResult]:
        """批量处理一个目录下所有 video review 目录。"""
        results = []
        for subdir in sorted(review_base_dir.iterdir()):
            if not subdir.is_dir():
                continue
            pairs_json = subdir / "pairs.json"
            if not pairs_json.exists():
                continue
            logger.info(f"AI 细筛: {subdir.name}")
            try:
                r = self.screen_review_package(subdir)
                results.append(r)
            except Exception as e:
                logger.error(f"AI 细筛失败 {subdir.name}: {e}")
        return results
