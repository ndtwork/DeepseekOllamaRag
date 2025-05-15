import requests

url = "http://192.168.5.179:11434/api/infer"
data = {
    "prompt": "code for me a simple calculator in python",
    "max_tokens": 100
}

response = requests.post(url, json=data)

# Kiểm tra kết quả
print(response.json())
