"""
FloodSense - IoT + ML Flood Prediction System
Main Flask API Server
"""
import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import numpy as np
from twilio.rest import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── Configuration ────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "ACb43a91b11a2cf6938a91403cfd7764ae")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",  "2f011b773a8b0cd3f32f73734db84a75")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+918532025050")
ALERT_PHONE_NUMBERS = os.getenv("ALERT_PHONE_NUMBERS", "+918532025050").split(",")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ─── Load ML Artifacts ────────────────────────────────────────────────────────
def load_artifacts():
    artifacts = {}
    try:
        artifacts["best_model"]    = joblib.load(f"{MODEL_DIR}/best_model.pkl")
        artifacts["rf_model"]      = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
        artifacts["gb_model"]      = joblib.load(f"{MODEL_DIR}/gb_model.pkl")
        artifacts["lr_model"]      = joblib.load(f"{MODEL_DIR}/lr_model.pkl")
        artifacts["scaler"]        = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        artifacts["le_land_cover"] = joblib.load(f"{MODEL_DIR}/le_land_cover.pkl")
        artifacts["le_soil_type"]  = joblib.load(f"{MODEL_DIR}/le_soil_type.pkl")
        artifacts["feature_cols"]  = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
        with open(f"{MODEL_DIR}/label_classes.json") as f:
            artifacts["label_classes"] = json.load(f)
        with open(f"{MODEL_DIR}/model_results.json") as f:
            artifacts["model_results"] = json.load(f)
        logger.info("✅ All ML artifacts loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {e}")
    return artifacts

ARTIFACTS = load_artifacts()

# Latest sensor readings (in-memory store, use Redis/DB in production)
latest_readings = {}
prediction_history = []

# ─── Helper Functions ─────────────────────────────────────────────────────────
def preprocess_input(data: dict) -> np.ndarray:
    """Convert raw sensor data to model-ready feature vector."""
    le_lc = ARTIFACTS["le_land_cover"]
    le_st = ARTIFACTS["le_soil_type"]

    land_cover = data.get("land_cover", "Agricultural")
    soil_type  = data.get("soil_type",  "Loam")

    # Handle unseen labels gracefully
    lc_classes = list(le_lc.classes_)
    st_classes  = list(le_st.classes_)
    lc_enc = le_lc.transform([land_cover])[0] if land_cover in lc_classes else 0
    st_enc  = le_st.transform([soil_type])[0]  if soil_type  in st_classes  else 2

    feature_vector = [
        float(data.get("rainfall_mm",              0)),
        float(data.get("temperature_c",            25)),
        float(data.get("humidity_pct",             60)),
        float(data.get("river_discharge_m3s",      0)),
        float(data.get("water_level_m",            0)),
        float(data.get("elevation_m",              300)),
        float(lc_enc),
        float(st_enc),
        float(data.get("infrastructure",           0)),
        float(data.get("historical_floods",        0)),
        float(data.get("barometric_pressure_hpa",  1013.25)),
        float(data.get("soil_moisture_pct",        40)),
    ]
    return np.array(feature_vector).reshape(1, -1)


def send_sms_alert(probability: float, location: str = "Uttarakhand"):
    """Send SMS alert via Twilio."""
    if TWILIO_ACCOUNT_SID == "your_account_sid":
        logger.warning("⚠️  Twilio not configured - SMS not sent (demo mode)")
        return {"status": "demo", "message": "Twilio not configured"}

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        risk_level = "HIGH" if probability >= 0.7 else "MODERATE"
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message_body = (
            f"🚨 FLOODSENSE ALERT [{risk_level} RISK]\n"
            f"Location: {location}\n"
            f"Flood Probability: {probability*100:.1f}%\n"
            f"Time: {timestamp}\n"
            f"⚠️ Take precautionary measures immediately!"
        )

        sent = []
        for number in ALERT_PHONE_NUMBERS:
            msg = client.messages.create(
                body=message_body,
                from_=TWILIO_FROM_NUMBER,
                to=number.strip()
            )
            sent.append({"to": number, "sid": msg.sid})
            logger.info(f"📱 SMS sent to {number}: {msg.sid}")

        return {"status": "sent", "messages": sent}
    except Exception as e:
        logger.error(f"❌ SMS error: {e}")
        return {"status": "error", "error": str(e)}


def get_risk_level(probability: float) -> dict:
    if probability >= 0.75:
        return {"level": "HIGH",     "color": "#ef4444", "action": "Evacuate immediately"}
    elif probability >= 0.50:
        return {"level": "MODERATE", "color": "#f59e0b", "action": "Stay alert, prepare to evacuate"}
    elif probability >= 0.25:
        return {"level": "LOW",      "color": "#3b82f6", "action": "Monitor conditions closely"}
    else:
        return {"level": "SAFE",     "color": "#22c55e", "action": "No immediate action required"}


# ─── API Routes ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "FloodSense API v1.0", "status": "running",
                    "endpoints": ["/predict", "/sensor-data", "/history", "/models/compare", "/health"]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "models_loaded": len(ARTIFACTS) > 0,
                    "timestamp": datetime.utcnow().isoformat()})


@app.route("/sensor-data", methods=["POST"])
def receive_sensor_data():
    """Endpoint for Arduino/IoT device to POST sensor readings."""
    try:
        data = request.get_json(force=True)
        data["received_at"] = datetime.utcnow().isoformat()
        latest_readings.update(data)
        logger.info(f"📡 Sensor data received: {data}")
        return jsonify({"status": "received", "data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint."""
    try:
        data = request.get_json(force=True)
        
        # Preprocess
        X = preprocess_input(data)
        X_scaled = ARTIFACTS["scaler"].transform(X)

        # Get predictions from all models
        models_map = {
            "Random Forest":      ARTIFACTS["rf_model"],
            "Gradient Boosting":  ARTIFACTS["gb_model"],
            "Logistic Regression": ARTIFACTS["lr_model"],
            "Best Model":         ARTIFACTS["best_model"],
        }

        all_preds = {}
        probabilities = []
        for name, model in models_map.items():
            prob = model.predict_proba(X_scaled)[0][1]
            pred = int(model.predict(X_scaled)[0])
            all_preds[name] = {"prediction": pred, "probability": round(prob, 4)}
            probabilities.append(prob)

        # Ensemble probability (average)
        ensemble_prob = float(np.mean(probabilities))
        flood_predicted = ensemble_prob >= 0.5
        risk = get_risk_level(ensemble_prob)

        # Build result
        result = {
            "timestamp":        datetime.utcnow().isoformat(),
            "sensor_data":      data,
            "flood_predicted":  flood_predicted,
            "ensemble_probability": round(ensemble_prob, 4),
            "risk_level":       risk["level"],
            "risk_color":       risk["color"],
            "action":           risk["action"],
            "model_predictions": all_preds,
        }

        # Store history
        prediction_history.append(result)
        if len(prediction_history) > 100:
            prediction_history.pop(0)

        # Update latest readings
        latest_readings.update(data)
        latest_readings["last_prediction"] = result

        # Send SMS if flood risk is HIGH or MODERATE
        sms_status = {"status": "not_sent", "reason": "risk_below_threshold"}
        if ensemble_prob >= 0.5:
            location = data.get("location", "Uttarakhand")
            sms_status = send_sms_alert(ensemble_prob, location)
            result["sms_alert"] = sms_status

        logger.info(f"🔮 Prediction: flood={flood_predicted}, prob={ensemble_prob:.3f}, risk={risk['level']}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/sensor", methods=["GET"])
def predict_from_latest_sensor():
    """Predict using last received sensor data."""
    if not latest_readings:
        return jsonify({"error": "No sensor data received yet"}), 404
    request._cached_json = (latest_readings, request._cached_json[1] if hasattr(request, '_cached_json') else None)
    # Manually call predict logic
    from flask import Request
    return predict()


@app.route("/history", methods=["GET"])
def get_history():
    limit = int(request.args.get("limit", 20))
    return jsonify({"history": prediction_history[-limit:], "total": len(prediction_history)})


@app.route("/latest", methods=["GET"])
def get_latest():
    return jsonify({"latest_readings": latest_readings})


@app.route("/models/compare", methods=["GET"])
def compare_models():
    return jsonify(ARTIFACTS.get("model_results", {}))


@app.route("/models/info", methods=["GET"])
def models_info():
    return jsonify({
        "available_models": ["Random Forest", "Gradient Boosting", "Logistic Regression", "Best Model (Ensemble)"],
        "features": ARTIFACTS.get("feature_cols", []),
        "label_classes": ARTIFACTS.get("label_classes", {}),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"🚀 FloodSense API starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
