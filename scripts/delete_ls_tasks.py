import requests

LABEL_STUDIO_URL = "http://localhost:8080"
TOKEN = "6eb6ce63c0d1e27948e73912a0816c620ff11aa6"

START_ID = 48106
END_ID = 48108

headers = {
    "Authorization": f"Token {TOKEN}"
}

for task_id in range(START_ID, END_ID + 1):
    url = f"{LABEL_STUDIO_URL}/api/tasks/{task_id}"
    r = requests.delete(url, headers=headers)

    if r.status_code in [200, 204]:
        print(f"Deleted task {task_id}")
    else:
        print(f"Failed {task_id}: {r.status_code} - {r.text}")