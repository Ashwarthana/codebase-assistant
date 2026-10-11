import urllib.request
import urllib.error
import json
import time

API_URL = "http://127.0.0.1:8000"
REPO_URL = "https://github.com/miguelgrinberg/flasky.git"

def req(url, method="GET", data=None):
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(data).encode("utf-8")
    with urllib.request.urlopen(req, data=data) as response:
        return json.loads(response.read().decode())

print("1. Indexing repository...")
res = req(f"{API_URL}/index", "POST", {"repo_url": REPO_URL})
job_id = res["job_id"]
print(f"Job ID: {job_id}")

print("Waiting for indexing to complete...")
while True:
    status_res = req(f"{API_URL}/index/status/{job_id}")
    status = status_res["status"]
    print(f"Current status: {status}")
    if status == "done":
        break
    elif status.startswith("error"):
        print("Indexing failed!")
        exit(1)
    time.sleep(5)

print("Indexing complete.\n")

questions = [
    "How does the login function authenticate a user?",
    "What is the purpose of config.py in this project?",
    "How does this application connect to MongoDB?"
]

print("2. Asking questions...")
for q in questions:
    print("-" * 40)
    print(f"Q: {q}")
    data = req(f"{API_URL}/ask", "POST", {"question": q, "mode": "proposed"})
    print(f"STATUS: {data['status']}")
    print("Preview of response:")
    answer = data.get("answer", "")
    print(answer[:150] + ("..." if len(answer) > 150 else ""))
print("-" * 40)
print("Smoke test complete.")
