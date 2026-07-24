"""Launch a persistent browser so you can log in to Douyin manually.
Session data (cookies, localStorage) is saved to browser_data/ for reuse.
"""
import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright

USER_DATA = Path(__file__).parent / "browser_data"
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA),
            headless=False,
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        )

        page = browser.pages[0] if browser.pages else await browser.new_page()
        await page.goto("https://www.douyin.com", wait_until="domcontentloaded", timeout=30000)

        print("浏览器已打开，请手动登录抖音。")
        print("登录完成后，在此输入 quit 并按 Enter 退出。")
        print(f"(Session 保存到 {USER_DATA})")

        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line.strip().lower() == "quit":
                break

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
