import json
import urllib.request


def main() -> None:
    payload = {
        "image": [[[128, 128, 128] for _ in range(32)] for _ in range(32)],
        "top_k": 3,
    }
    request = urllib.request.Request(
        "http://localhost:8000/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
