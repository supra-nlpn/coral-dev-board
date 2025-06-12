import sys
import requests

if len(sys.argv) != 2:
    print("Usage: python3 client.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
url = 'http://10.245.86.157:8000/predict'  # Dev Board  IP address and port

with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

try:
    print(response.json())
except Exception as e:
    print("Error parsing JSON response:", e)
    print("Raw response:", response.text)

