import serial
import requests
import json

SERIAL_PORT = "COM10"   # change if needed
BAUD_RATE = 9600

API_URL = "https://floodsense-4.onrender.com/predict"

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

print("Listening to Arduino...")

while True:
    line = ser.readline().decode().strip()

    if line.startswith("DATA:"):
        try:
            data = json.loads(line[5:])
            print("Sending:", data)

            res = requests.post(API_URL, json=data)
            print("Response:", res.json())

        except Exception as e:
            print("Error:", e)