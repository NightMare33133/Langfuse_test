import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()


API_BASE = "http://localhost/v1"
API_KEY = os.getenv("DIFY_API_KEY")

questions = [
    "AISP 和 PISP 的区别是什么？",
    "Bank 1.0 到 Bank 4.0 的区别是什么？",
    "Digitization、Digitalization 和 Digital Transformation 的区别是什么？",
    "Fintech ecosystem players 包括哪些角色？",
    "Fintech 的六大 verticals 是什么？",
    "Fintech 的定义是什么？",
    "P2P Lending 是如何运作的？",
    "什么是 Neo Bank？它的特点是什么？",
    "什么是 Open Banking？",
    "什么是 The Chasm？",
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

results = []

for q in questions:
    payload = {
        "inputs": {},
        "query": q,
        "response_mode": "blocking",
        "user": "batch-test-001"
    }

    try:
        resp = requests.post(
            f"{API_BASE}/chat-messages",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        results.append({
            "question": q,
            "response": data,
        })
        print(f"OK: {q}")
    except Exception as e:
        results.append({
            "question": q,
            "error": str(e),
        })
        print(f"ERROR: {q} -> {e}")

    time.sleep(1)

with open("batch_results.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done.")