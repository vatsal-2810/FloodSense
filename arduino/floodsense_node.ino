/*
 * ============================================================
 *  FloodSense IoT Node — Arduino + ESP8266/ESP32
 *  Sensors: Rainfall, Water Level, Soil Moisture,
 *           DHT22 (Temp/Humidity), BMP280 (Barometric Pressure)
 * ============================================================
 *
 *  WIRING DIAGRAM (Summary):
 *  ──────────────────────────────────────────────────────────
 *  RAINFALL SENSOR (YL-83 / FC-37):
 *    VCC  → 3.3V
 *    GND  → GND
 *    AO   → A0  (analog output, 0-1023)
 *    DO   → D7  (digital trigger, threshold-based)
 *
 *  WATER LEVEL SENSOR (Ultrasonic HC-SR04):
 *    VCC  → 5V
 *    GND  → GND
 *    TRIG → D5
 *    ECHO → D6
 *
 *  SOIL MOISTURE SENSOR (Capacitive v1.2):
 *    VCC  → 3.3V
 *    GND  → GND
 *    AOUT → A1  (analog, if using ESP32 with multiple ADC pins)
 *    Note: Use voltage divider if powering from 5V on Arduino Uno A0
 *
 *  DHT22 (Temperature + Humidity):
 *    VCC  → 3.3V/5V
 *    GND  → GND
 *    DATA → D4  (with 10kΩ pull-up to VCC)
 *
 *  BMP280 (Barometric Pressure) via I2C:
 *    VCC  → 3.3V
 *    GND  → GND
 *    SCL  → D1 (I2C Clock)
 *    SDA  → D2 (I2C Data)
 *    CSB  → 3.3V (I2C mode)
 *    SDO  → GND (Address 0x76) or 3.3V (Address 0x77)
 *
 *  ESP8266 (NodeMCU) / ESP32 WiFi Module:
 *    If using Arduino Uno + separate ESP8266:
 *      ESP TX → Arduino RX (pin 0) via 3.3V logic level shifter
 *      ESP RX → Arduino TX (pin 1) via voltage divider (5V→3.3V)
 *    Recommended: Use NodeMCU ESP8266 directly (all-in-one)
 *
 *  POWER:
 *    Arduino Uno/Nano: USB or 9V barrel
 *    NodeMCU / ESP32:  USB or LiPo 3.7V with boost converter
 * ──────────────────────────────────────────────────────────
 *
 *  LIBRARIES REQUIRED (install via Arduino Library Manager):
 *    - DHT sensor library by Adafruit
 *    - Adafruit BMP280 Library
 *    - Adafruit Unified Sensor
 *    - ArduinoJson (v6.x)
 *    - ESP8266WiFi (built-in for NodeMCU) OR WiFi.h (ESP32)
 *    - ESP8266HTTPClient OR HTTPClient (ESP32)
 *
 * ============================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>
#include <Adafruit_BMP280.h>
#include <ArduinoJson.h>

// ── WiFi & HTTP (choose one board) ──
#ifdef ESP32
  #include <WiFi.h>
  #include <HTTPClient.h>
#else
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
  #include <WiFiClient.h>
#endif

// ── WiFi Credentials ──
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ── FloodSense Cloud API ──
const char* API_BASE_URL   = "https://floodsense-api.onrender.com";  // your Render URL
const char* SENSOR_ENDPOINT = "/sensor-data";
const char* PREDICT_ENDPOINT = "/predict";

// ── Pin Definitions ──
#define RAINFALL_ANALOG_PIN  A0
#define RAINFALL_DIGITAL_PIN D7
#define ULTRASONIC_TRIG_PIN  D5
#define ULTRASONIC_ECHO_PIN  D6
#define SOIL_MOISTURE_PIN    A1    // ESP32 uses GPIO34/35 for second ADC
#define DHT_PIN              D4
#define DHT_TYPE             DHT22

// ── Sensor Objects ──
DHT           dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP280 bmp;

// ── Configuration ──
const char* LOCATION       = "Dehradun, Uttarakhand";
const char* LAND_COVER     = "Agricultural";    // Update for your site
const char* SOIL_TYPE      = "Loam";            // Update for your site
const int   INFRASTRUCTURE = 1;                 // 1 = present, 0 = absent
const int   HISTORICAL_FLOODS = 0;              // 1 = yes, 0 = no
const float SENSOR_ELEVATION_M = 300.0;         // Site elevation in meters

// Transmission interval (30 seconds)
const unsigned long TRANSMIT_INTERVAL_MS = 30000;
unsigned long lastTransmitTime = 0;

// ── Calibration Constants ──
const int RAINFALL_AIR_VALUE    = 1023;  // Sensor reading in dry air
const int RAINFALL_WATER_VALUE  = 200;   // Sensor reading submerged
const int SOIL_DRY_VALUE        = 890;   // Sensor in dry soil
const int SOIL_WET_VALUE        = 380;   // Sensor in saturated soil
const float MAX_WATER_LEVEL_CM  = 500.0; // HC-SR04 max range in cm

// ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println(F("\n🌊 FloodSense IoT Node Starting..."));

  // DHT22
  dht.begin();
  Serial.println(F("✅ DHT22 initialized"));

  // BMP280 (I2C)
  Wire.begin();
  if (!bmp.begin(0x76)) {
    Serial.println(F("❌ BMP280 not found! Check wiring. Trying 0x77..."));
    if (!bmp.begin(0x77)) {
      Serial.println(F("❌ BMP280 failed on both addresses. Check connections."));
    }
  } else {
    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                    Adafruit_BMP280::SAMPLING_X2,
                    Adafruit_BMP280::SAMPLING_X16,
                    Adafruit_BMP280::FILTER_X16,
                    Adafruit_BMP280::STANDBY_MS_500);
    Serial.println(F("✅ BMP280 initialized"));
  }

  // Pin modes
  pinMode(RAINFALL_DIGITAL_PIN, INPUT);
  pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);

  // Connect WiFi
  connectWiFi();
}

// ─────────────────────────────────────────────────────────────────
void loop() {
  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("⚠️  WiFi lost. Reconnecting..."));
    connectWiFi();
  }

  unsigned long now = millis();
  if (now - lastTransmitTime >= TRANSMIT_INTERVAL_MS) {
    lastTransmitTime = now;
    
    // Read all sensors
    SensorData data = readAllSensors();
    printSensorData(data);

    // Send to cloud
    if (WiFi.status() == WL_CONNECTED) {
      String payload = buildJSON(data);
      bool sent = sendToCloud(payload);
      if (sent) {
        Serial.println(F("✅ Data sent to FloodSense Cloud"));
      }
    }
  }

  delay(100);
}

// ─────────────────────────────────────────────────────────────────
struct SensorData {
  float rainfall_mm;
  float temperature_c;
  float humidity_pct;
  float water_level_m;
  float barometric_pressure_hpa;
  float soil_moisture_pct;
  // Derived / fixed
  float river_discharge_m3s;
  float elevation_m;
  String land_cover;
  String soil_type;
  int   infrastructure;
  int   historical_floods;
};

// ── Read Rainfall ──
float readRainfall() {
  int raw = analogRead(RAINFALL_ANALOG_PIN);
  // Map analog value to mm (approximate calibration)
  // Higher moisture = lower analog value
  float pct = map(raw, RAINFALL_WATER_VALUE, RAINFALL_AIR_VALUE, 100, 0);
  pct = constrain(pct, 0, 100);
  // Convert percentage to rough mm (0-300mm range for Uttarakhand monsoon)
  return pct * 3.0;
}

// ── Read Water Level via HC-SR04 ──
float readWaterLevel() {
  // Send 10µs pulse
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);

  long duration = pulseIn(ULTRASONIC_ECHO_PIN, HIGH, 30000); // 30ms timeout
  if (duration == 0) return -1.0; // No echo

  float distance_cm = (duration * 0.0343) / 2.0;
  // Water level = sensor height above empty tank - measured distance
  // Adjust SENSOR_HEIGHT_CM to match your installation
  const float SENSOR_HEIGHT_CM = 200.0;
  float water_cm = SENSOR_HEIGHT_CM - distance_cm;
  water_cm = constrain(water_cm, 0, MAX_WATER_LEVEL_CM);
  return water_cm / 100.0; // Return in meters
}

// ── Read Soil Moisture ──
float readSoilMoisture() {
  int raw = analogRead(SOIL_MOISTURE_PIN);
  float pct = map(raw, SOIL_DRY_VALUE, SOIL_WET_VALUE, 0, 100);
  return constrain(pct, 0, 100);
}

// ── Read All Sensors ──
SensorData readAllSensors() {
  SensorData d;

  // DHT22
  d.temperature_c  = dht.readTemperature();
  d.humidity_pct   = dht.readHumidity();
  if (isnan(d.temperature_c)) d.temperature_c = 25.0; // fallback
  if (isnan(d.humidity_pct))  d.humidity_pct  = 60.0;

  // BMP280
  d.barometric_pressure_hpa = bmp.readPressure() / 100.0F; // Pa → hPa

  // Custom sensors
  d.rainfall_mm        = readRainfall();
  d.water_level_m      = readWaterLevel();
  d.soil_moisture_pct  = readSoilMoisture();

  // Derived / site-specific constants
  // River discharge: estimated from water level (simplified rating curve)
  // Q = a * (H - H0)^b  — use actual gauging station data if available
  d.river_discharge_m3s = (d.water_level_m > 0) ? (5.2 * pow(d.water_level_m, 1.67)) : 0;
  
  d.elevation_m      = SENSOR_ELEVATION_M;
  d.land_cover       = String(LAND_COVER);
  d.soil_type        = String(SOIL_TYPE);
  d.infrastructure   = INFRASTRUCTURE;
  d.historical_floods = HISTORICAL_FLOODS;

  return d;
}

// ── Build JSON Payload ──
String buildJSON(SensorData& d) {
  StaticJsonDocument<512> doc;

  doc["rainfall_mm"]              = round(d.rainfall_mm * 100) / 100.0;
  doc["temperature_c"]            = round(d.temperature_c * 100) / 100.0;
  doc["humidity_pct"]             = round(d.humidity_pct * 100) / 100.0;
  doc["river_discharge_m3s"]      = round(d.river_discharge_m3s * 100) / 100.0;
  doc["water_level_m"]            = round(d.water_level_m * 100) / 100.0;
  doc["elevation_m"]              = d.elevation_m;
  doc["barometric_pressure_hpa"]  = round(d.barometric_pressure_hpa * 100) / 100.0;
  doc["soil_moisture_pct"]        = round(d.soil_moisture_pct * 100) / 100.0;
  doc["land_cover"]               = d.land_cover;
  doc["soil_type"]                = d.soil_type;
  doc["infrastructure"]           = d.infrastructure;
  doc["historical_floods"]        = d.historical_floods;
  doc["location"]                 = LOCATION;

  String payload;
  serializeJson(doc, payload);
  return payload;
}

// ── Send Data to Cloud ──
bool sendToCloud(const String& payload) {
  WiFiClient wifiClient;
  HTTPClient http;

  String url = String(API_BASE_URL) + PREDICT_ENDPOINT;
  http.begin(wifiClient, url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000); // 10s timeout

  int code = http.POST(payload);
  
  if (code > 0) {
    if (code == HTTP_CODE_OK || code == 201) {
      String response = http.getString();
      Serial.print(F("📡 API Response ["));
      Serial.print(code);
      Serial.println(F("]:"));
      Serial.println(response.substring(0, 200)); // Print first 200 chars

      // Parse flood prediction from response
      StaticJsonDocument<1024> resp;
      DeserializationError err = deserializeJson(resp, response);
      if (!err) {
        bool flooded     = resp["flood_predicted"].as<bool>();
        float prob       = resp["ensemble_probability"].as<float>();
        const char* risk = resp["risk_level"];
        Serial.printf("🔮 Flood Predicted: %s | Probability: %.1f%% | Risk: %s\n",
                      flooded ? "YES ⚠️" : "NO ✅", prob * 100, risk);
        
        // Optional: drive local LED or buzzer based on prediction
        // if (flooded) { digitalWrite(BUZZER_PIN, HIGH); }
      }
      http.end();
      return true;
    }
  } else {
    Serial.printf("❌ HTTP error: %s\n", http.errorToString(code).c_str());
  }
  http.end();
  return false;
}

// ── Connect WiFi ──
void connectWiFi() {
  Serial.printf("📶 Connecting to WiFi: %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n✅ WiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println(F("\n❌ WiFi connection failed. Will retry next cycle."));
  }
}

// ── Print Sensor Readings to Serial Monitor ──
void printSensorData(SensorData& d) {
  Serial.println(F("\n─────────── FloodSense Sensor Reading ───────────"));
  Serial.printf("🌧  Rainfall:          %.2f mm\n",      d.rainfall_mm);
  Serial.printf("🌡  Temperature:       %.2f °C\n",      d.temperature_c);
  Serial.printf("💧 Humidity:           %.1f %%\n",      d.humidity_pct);
  Serial.printf("🌊 Water Level:        %.3f m\n",       d.water_level_m);
  Serial.printf("🏞  River Discharge:   %.2f m³/s\n",    d.river_discharge_m3s);
  Serial.printf("🧱 Soil Moisture:      %.1f %%\n",      d.soil_moisture_pct);
  Serial.printf("🌬  Pressure:          %.2f hPa\n",     d.barometric_pressure_hpa);
  Serial.println(F("──────────────────────────────────────────────────"));
}
