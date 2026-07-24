"""测试 AI API 连通性和可用模型（OpenAI 兼容接口）。

用法:
    set AI_API_KEY=your_key        (Windows)
    set AI_BASE_URL=https://your-api.com/v1
    python test_mimo_api.py
"""
import json
import os
import urllib.request
import urllib.error

BASE_URL = os.environ.get("AI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("AI_API_KEY", "")

def test_models():
    url = f"{BASE_URL}/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            models = body.get("data", [])
            print(f"[OK] Available models ({len(models)}):")
            for m in models:
                mid = m.get("id", "unknown")
                print(f"  - {mid}")
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code}]", e.read().decode("utf-8")[:500])
    except Exception as e:
        print(f"[ERROR] {e}")

def test_text():
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 10,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            print("[OK] Text reply:", body["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code}]", e.read().decode("utf-8")[:300])
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("[TEST] List models")
    test_models()
    print()
    print("=" * 50)
    print("[TEST] Text completion")
    test_text()
