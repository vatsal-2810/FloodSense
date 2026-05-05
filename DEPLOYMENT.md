# FloodSense — Deployment Guide
## IoT + ML Flood Prediction System

---

## 🏗️ Project Architecture

```
[Arduino/NodeMCU IoT Sensors]
        │ POST JSON via WiFi (HTTP)
        ▼
[FloodSense Cloud API — Flask on Render.com (FREE)]
        │
        ├─→ ML Models (Random Forest, GB, SVM, LR, LSTM-inspired)
        │         └─→ Ensemble Prediction
        │
        ├─→ Twilio SMS Alert (if flood risk ≥ 50%)
        │
        └─→ Dashboard UI (React/HTML frontend)
```

---

## 📡 Sensor Wiring Quick Reference

| Sensor | VCC | GND | Signal Pin(s) |
|--------|-----|-----|---------------|
| Rainfall (YL-83) | 3.3V | GND | A0 (AO), D7 (DO) |
| Water Level (HC-SR04) | 5V | GND | D5 (TRIG), D6 (ECHO) |
| Soil Moisture (Capacitive) | 3.3V | GND | A1 (AOUT) |
| DHT22 Temp/Humidity | 3.3V | GND | D4 + 10kΩ pull-up |
| BMP280 Pressure (I2C) | 3.3V | GND | D1 (SCL), D2 (SDA) |

**Recommended board:** NodeMCU ESP8266 or ESP32 (has WiFi built-in).
For Arduino Uno: Add ESP8266-01 module on UART pins with voltage level shifter.

---

## ☁️ FREE Cloud Deployment — Render.com

### Step 1: Prepare your GitHub repository

```bash
# Initialize git in your FloodSense folder
git init
git add .
git commit -m "FloodSense initial commit"

# Push to GitHub
gh repo create floodsense --public --push
# OR manually create repo at github.com and push
```

### Step 2: Deploy on Render (100% Free)

1. Go to **https://render.com** → Sign up (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo: `floodsense`
4. Configure:
   - **Name:** `floodsense-api`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`
   - **Plan:** `Free`
5. Add **Environment Variables** (click "Environment"):
   ```
   TWILIO_ACCOUNT_SID = ACxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN  = xxxxxxxxxxxxxxxx
   TWILIO_FROM_NUMBER = +1XXXXXXXXXX
   ALERT_PHONE_NUMBERS = +919XXXXXXXXX
   ```
6. Click **"Create Web Service"** → Wait ~3 minutes for deploy
7. Your API URL: `https://floodsense-api.onrender.com`

> ⚠️ **Free tier note:** Render free instances spin down after 15 min of inactivity.
> First request after idle takes ~30s. For always-on, upgrade to $7/mo Starter.
> Alternative always-free: **Railway.app** (500 hrs/month free)

### Step 3: Update Arduino code

In `floodsense_node.ino`, replace:
```cpp
const char* API_BASE_URL = "https://floodsense-api.onrender.com";
```
with your actual Render URL.

---

## 📱 Twilio SMS Setup (Free Trial)

1. Sign up at **https://twilio.com** (free $15 trial credit)
2. Go to **Console Dashboard**
3. Note your:
   - **Account SID** (starts with AC)
   - **Auth Token**
4. Get a phone number: **Phone Numbers → Buy a Number** (free with trial)
5. Add environment variables to Render (above)

> Trial limitation: Can only send SMS to verified numbers. Go to
> **Verified Caller IDs** and add your phone number to receive alerts.

---

## 🧪 Testing the API

### Test prediction (curl):
```bash
curl -X POST https://floodsense-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_mm": 280,
    "temperature_c": 22,
    "humidity_pct": 95,
    "river_discharge_m3s": 450,
    "water_level_m": 3.5,
    "elevation_m": 300,
    "barometric_pressure_hpa": 985,
    "soil_moisture_pct": 88,
    "land_cover": "Agricultural",
    "soil_type": "Clay",
    "infrastructure": 1,
    "historical_floods": 1,
    "location": "Dehradun, Uttarakhand"
  }'
```

### Expected response:
```json
{
  "flood_predicted": true,
  "ensemble_probability": 0.87,
  "risk_level": "HIGH",
  "action": "Evacuate immediately",
  "sms_alert": {"status": "sent"},
  "model_predictions": { ... }
}
```

### Check health:
```bash
curl https://floodsense-api.onrender.com/health
```

### Compare models:
```bash
curl https://floodsense-api.onrender.com/models/compare
```

---

## 🔄 Alternative Free Cloud Options

| Platform | Free Tier | Notes |
|----------|-----------|-------|
| **Render.com** ✅ | Always free (sleeps) | Best for Flask |
| **Railway.app** ✅ | 500 hrs/month | Fast cold starts |
| **Fly.io** | 3 shared VMs free | More complex setup |
| **PythonAnywhere** | 1 web app free | No custom domains on free |
| **Koyeb** | 2 free services | Good for EU region |

---

## 📊 Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (already done, models/ folder exists)
python train_models.py

# Run API locally
python app.py

# API available at: http://localhost:5000
```

---

## 🔁 Data Flow Summary

```
1. Arduino reads sensors every 30 seconds
2. Sends JSON POST to → /predict endpoint
3. API preprocesses → runs 4 ML models → ensemble vote
4. Returns: flood_predicted, probability, risk_level
5. If probability ≥ 0.50 → Twilio sends SMS to alert numbers
6. Dashboard polls /latest and /history for visualization
```

---

## 📁 File Structure

```
FloodSense/
├── app.py                    ← Flask API server
├── train_models.py           ← ML training script
├── requirements.txt          ← Python dependencies
├── Procfile                  ← Gunicorn start command
├── render.yaml               ← Render deployment config
├── .env.example              ← Environment variable template
├── flood_dataset_cleaned.csv ← Training data
├── models/                   ← Saved ML artifacts
│   ├── best_model.pkl
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   ├── le_land_cover.pkl
│   ├── le_soil_type.pkl
│   ├── feature_cols.pkl
│   ├── model_results.json
│   └── label_classes.json
└── arduino/
    └── floodsense_node.ino   ← Arduino firmware
```
