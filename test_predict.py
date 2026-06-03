import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "engine_id": "ENG-777",
    "s11": 1.0,
    "s12": 2.0,
    "s13": 3.0,
    "s15": 4.0,
    "lat": 18.5204,
    "lon": 73.8567
}

for i in range(4):
    print(f"Request {i+1}")
    res = requests.post(url, json=data)
    print(res.status_code)
    try:
        print(res.json())
    except:
        print(res.text)
