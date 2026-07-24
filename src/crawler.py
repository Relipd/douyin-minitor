"""Douyin video crawler using Playwright browser automation."""
import asyncio
import random
import sys
from pathlib import Path
from playwright.async_api import async_playwright


class DouyinCrawler:
    def __init__(self, headless=True, user_data_dir=None):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._context = None
        self._browser = None

    # ── browser lifecycle ──────────────────────────────────────────

    async def start(self, use_persistent_context=False):
        """启动浏览器。

        Args:
            use_persistent_context: 是否使用持久化上下文（搜索模式需要登录态才启用）。
                                    直链模式无需登录，用普通 launch 更稳定。
        """
        await self.close()
        self._playwright = await async_playwright().start()
        if use_persistent_context and self.user_data_dir:
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir,
                headless=self.headless,
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            )
        else:
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            )

    async def restart(self, use_persistent_context=False):
        """Re-create the browser context after a crash or manual close."""
        print("[CRAWLER] Restarting browser...")
        await self.close()
        await self.start(use_persistent_context=use_persistent_context)

    async def close(self):
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

    def _ensure_context(self):
        if not self._context:
            raise RuntimeError("Crawler not started — call await crawler.start() first")

    # ── CAPTCHA ────────────────────────────────────────────────────

    async def _wait_for_verify(self, page, timeout: int = 120) -> bool:
        """检测验证码并等待人工完成。

        Args:
            page: Playwright 页面对象。
            timeout: 等待人工完成的超时秒数（默认 120 秒）。
                     超时后自动跳过，返回 True（不阻塞流程）。
        """
        verify_selectors = [
            ".captcha_verify_container",
            ".verify-captcha",
            "#captcha-verify-image",
            ".security-verify",
            ".douyin-captcha",
            "[class*='captcha']",
            "[class*='verify']",
            "text=/请完成.*验证/",
            "text=/安全验证/",
            "text=/拖动.*滑块/",
            "text=/点击.*相同/",
        ]
        for sel in verify_selectors:
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    break
            except Exception:
                continue
        else:
            return False

        print(f"\n[!] 检测到人机验证，请在浏览器中手动完成验证（{timeout}s 超时自动跳过）。")
        print("[!] 完成后回到终端按 Enter 继续...")
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, sys.stdin.readline),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            print(f"\n[!] 验证码等待超时 ({timeout}s)，自动跳过继续执行。")
            return True
        await asyncio.sleep(2)
        return True

    # ── search ─────────────────────────────────────────────────────

    async def search(self, keyword: str, max_videos=30, scroll_count=5, pause=3) -> list[dict]:
        self._ensure_context()
        videos = []

        page = await self._context.new_page()
        try:
            search_url = f"https://www.douyin.com/search/{keyword}?type=video"
            await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(3)
            await self._wait_for_verify(page)

            for i in range(scroll_count):
                scroll_y = random.randint(300, 800)
                await page.evaluate(f"window.scrollBy(0, {scroll_y})")
                await asyncio.sleep(pause + random.uniform(1.0, 3.0))
                if i % 3 == 0:
                    await self._wait_for_verify(page)

            await page.wait_for_timeout(2000)
            video_elements = await page.query_selector_all(
                '[data-e2e="search-video-item"], .search-result-card, ul[class*="list"] > li'
            )
            if not video_elements:
                video_elements = await page.query_selector_all('a[href*="/video/"]')

            seen_ids = set()
            for el in video_elements:
                if len(videos) >= max_videos:
                    break
                try:
                    info = await self._parse_video_element(el)
                    if info and info["video_id"] not in seen_ids:
                        seen_ids.add(info["video_id"])
                        videos.append(info)
                except Exception:
                    continue

            if len(videos) < 5:
                videos = await self._extract_from_links(page, max_videos)
        finally:
            await page.close()

        return videos

    # ── frame capture from video page ──────────────────────────────

    async def _check_video_playing(self, page) -> bool:
        """Check if the video element is actually playing (not paused/stuck)."""
        try:
            state = await page.evaluate("""
                () => {
                    const v = document.querySelector('video');
                    if (!v) return {exists: false};
                    return {
                        exists: true,
                        paused: v.paused,
                        ended: v.ended,
                        currentTime: v.currentTime,
                        readyState: v.readyState,
                        duration: v.duration || 0,
                    };
                }
            """)
            if not state.get("exists", False):
                return False
            return (not state.get("paused", True)
                    and not state.get("ended", False)
                    and state.get("readyState", 0) >= 2
                    and state.get("currentTime", 0) > 0)
        except Exception:
            return False

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        """从抖音 URL 中提取视频 ID。"""
        if "/video/" in url:
            return url.split("/video/")[-1].split("?")[0].rstrip("/")
        return None

    async def _wait_for_playback(self, page, timeout=8) -> bool:
        """Wait for video to start playing, retrying play() if needed."""
        for _ in range(int(timeout / 0.5)):
            if await self._check_video_playing(page):
                return True
            # Retry play
            await page.evaluate("""
                () => {
                    const v = document.querySelector('video');
                    if (v) {
                        v.muted = true;
                        v.play().catch(() => {});
                    }
                }
            """)
            await asyncio.sleep(0.5)
        return False

    async def _frames_differ(self, page, prev_screenshot: bytes = None) -> tuple[bool, bytes]:
        """Take a screenshot and check if it differs from the previous one.

        Returns (is_different, current_screenshot_bytes).
        Used to detect if video is actually advancing frames.
        """
        try:
            video_el = await page.query_selector("video, .xgplayer video, [class*='player'] video")
            if video_el:
                screenshot = await video_el.screenshot()
            else:
                screenshot = await page.screenshot(full_page=False)

            if prev_screenshot is None:
                return True, screenshot

            # Simple byte-level difference check
            if len(screenshot) == len(prev_screenshot):
                diff_count = sum(1 for a, b in zip(screenshot, prev_screenshot) if a != b)
                diff_ratio = diff_count / len(screenshot)
                return diff_ratio > 0.005, screenshot  # 0.5% difference threshold
            return True, screenshot
        except Exception:
            return True, b""

    async def _get_video_duration(self, page) -> float:
        """读取视频总时长（秒），读取失败时返回 0 让调用方兜底。"""
        try:
            dur = await page.evaluate("""
                () => {
                    const v = document.querySelector('video');
                    if (!v) return 0;
                    return v.duration || 0;
                }
            """)
            return float(dur) if dur and float(dur) > 0 else 0
        except Exception:
            return 0

    async def capture_video_frames(
        self, video_url: str, output_dir: Path, num_frames=40, interval=0.5,
        random_start=False, retry_on_error=True, skip_verify=True
    ) -> list[Path]:
        """Open a Douyin video page, play the video, and take screenshots
        of the video element as it plays.

        截帧策略（v2 — 按视频时长自适应）：
          1. 先读取视频的实际 duration
          2. 按 interval 间隔从 0 到 duration 均匀截取
          3. 短视频少截、长视频多截，每帧间隔固定 0.5s
          4. 视频播放完后自动停止，不硬等

        Args:
            skip_verify: 跳过人机验证检测（直链模式无需验证，搜索模式需要）

        Args:
            num_frames: 仅作为上限保护（默认 40 → 最多 40 帧，防止过长视频爆帧）
            interval: 帧间隔秒数（默认 0.5s）
            random_start: 启用时在视频前 3s 随机偏移起始点
        """
        self._ensure_context()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            page = await self._context.new_page()
        except Exception:
            if retry_on_error:
                await self.restart(use_persistent_context=False)
                return await self.capture_video_frames(
                    video_url, output_dir, num_frames, interval,
                    random_start, retry_on_error=False, skip_verify=skip_verify)
            raise

        saved = []
        try:
            # ── 加载页面 ──
            try:
                resp = await page.goto(video_url, wait_until="domcontentloaded", timeout=60000)
                if resp and resp.status >= 400:
                    print(f"    [SKIP] 页面返回 {resp.status}（视频可能已删除），跳过")
                    return []
            except Exception as nav_err:
                print(f"    [SKIP] 页面加载失败: {nav_err}，跳过")
                return []

            # ── 等待 7 秒让页面加载/重定向完成 ──
            await asyncio.sleep(7)

            # ── URL 匹配验证：确认当前页确实是目标视频 ──
            current_url = page.url
            orig_vid = self._extract_video_id(video_url)
            curr_vid = self._extract_video_id(current_url)
            if not curr_vid or curr_vid != orig_vid:
                print(f"    [SKIP] URL 不匹配 (期望 {orig_vid}, 当前 {curr_vid or '非视频页'})，清理并跳过")
                # 清理已创建的帧目录
                if output_dir.exists():
                    import shutil
                    shutil.rmtree(output_dir, ignore_errors=True)
                return []

            # 检查页面中是否有 video 元素
            has_video = await page.evaluate("() => !!document.querySelector('video')")
            if not has_video:
                print(f"    [SKIP] 页面中未找到视频元素，跳过")
                return []

            # Start the video playing via JS
            await page.evaluate("""
                () => {
                    const v = document.querySelector('video');
                    if (v) {
                        v.muted = true;
                        v.play().catch(() => {});
                    }
                }
            """)

            # 读取视频时长
            duration = await self._get_video_duration(page)
            if duration <= 0:
                # 兜底：读不到时长就用固定帧数
                duration = num_frames * interval
                print(f"    [WARN] 未能读取视频时长，使用固定 {num_frames} 帧")

            # 按视频时长 + interval 计算帧数，但不超过 num_frames（上限保护）
            if duration > 0:
                computed = int(duration / interval)
                total_to_capture = min(computed, num_frames)
            else:
                total_to_capture = num_frames

            print(f"    [CAPTURE] 视频时长={duration:.1f}s → "
                  f"按 {interval}s/帧 应截 {int(duration/interval)} 帧, "
                  f"上限保护后={total_to_capture} 帧")

            if random_start:
                start_delay = random.uniform(0, 3.0)
                await asyncio.sleep(start_delay)
            else:
                await asyncio.sleep(1)

            # Wait for video to actually start playing
            try:
                playing = await self._wait_for_playback(page, timeout=6)
                if not playing:
                    print(f"    [WARN] Video may not be playing, capturing anyway")
            except Exception as play_err:
                print(f"    [SKIP] 视频播放失败: {play_err}，跳过")
                return []

            prev_screenshot = None
            consecutive_same = 0
            max_same = 5
            url_check_counter = 0     # 每 20 帧（≈10s）检测一次 URL 是否跳走

            for i in range(total_to_capture):
                frame_path = output_dir / f"frame_{i+1:04d}.jpg"
                try:
                    # 每 20 帧检测一次页面 URL 是否仍匹配
                    url_check_counter += 1
                    if url_check_counter >= 20:
                        url_check_counter = 0
                        try:
                            curr_url = page.url
                            curr_vid = self._extract_video_id(curr_url)
                            if not curr_vid or curr_vid != orig_vid:
                                print(f"    [SKIP] 播放中 URL 跳走 (期望 {orig_vid}, 当前 {curr_vid or '非视频页'})，跳过")
                                break
                        except Exception:
                            pass

                    # 检测视频是否已播放完毕
                    ended = await page.evaluate("""
                        () => {
                            const v = document.querySelector('video');
                            return v ? v.ended : false;
                        }
                    """)
                    if ended:
                        print(f"    [END] 视频播放完毕，共截 {len(saved)} 帧")
                        break

                    is_different, screenshot = await self._frames_differ(page, prev_screenshot)

                    if is_different or consecutive_same >= max_same:
                        video_el = await page.query_selector("video, .xgplayer video, [class*='player'] video")
                        if video_el:
                            await video_el.screenshot(path=str(frame_path))
                        else:
                            await page.screenshot(path=str(frame_path), full_page=False)
                        saved.append(frame_path)
                        prev_screenshot = screenshot
                        consecutive_same = 0
                    else:
                        consecutive_same += 1
                        if consecutive_same <= 2:
                            video_el = await page.query_selector("video, .xgplayer video, [class*='player'] video")
                            if video_el:
                                await video_el.screenshot(path=str(frame_path))
                            else:
                                await page.screenshot(path=str(frame_path), full_page=False)
                            saved.append(frame_path)
                except Exception:
                    pass

                actual_interval = interval
                if consecutive_same > 0:
                    actual_interval = interval * 0.5
                await asyncio.sleep(actual_interval)
        finally:
            await page.close()

        return saved

    # ── video element parsing ──────────────────────────────────────

    async def _parse_video_element(self, el) -> dict | None:
        try:
            href = await el.get_attribute("href")
            if not href:
                link = await el.query_selector("a")
                if link:
                    href = await link.get_attribute("href")

            if not href or "/video/" not in href:
                return None

            video_id = href.split("/video/")[-1].split("?")[0].rstrip("/")

            title_el = await el.query_selector("[class*='title'], .search-result-title, p, span")
            title = await title_el.inner_text() if title_el else ""
            title = title.strip()[:200]

            author_el = await el.query_selector("[class*='author'], [class*='name'], .nickname")
            author = await author_el.inner_text() if author_el else ""

            return {
                "video_id": video_id,
                "url": f"https://www.douyin.com/video/{video_id}",
                "title": title,
                "author": author,
            }
        except Exception:
            return None

    async def _extract_from_links(self, page, max_videos) -> list[dict]:
        links = await page.query_selector_all('a[href*="/video/"]')
        videos = []
        seen = set()
        seen_titles = set()

        for link in links:
            if len(videos) >= max_videos:
                break
            try:
                href = await link.get_attribute("href")
                if not href or "/video/" not in href:
                    continue
                video_id = href.split("/video/")[-1].split("?")[0].rstrip("/")
                if video_id in seen:
                    continue
                seen.add(video_id)

                text = await link.inner_text()
                text = text.strip()[:200]
                if text and text not in seen_titles:
                    seen_titles.add(text)

                videos.append({
                    "video_id": video_id,
                    "url": f"https://www.douyin.com/video/{video_id}",
                    "title": text,
                    "author": "",
                })
            except Exception:
                continue
        return videos
