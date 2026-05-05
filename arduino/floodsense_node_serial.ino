// ----------- Analog Pins -----
#define RAIN_PIN A0
#define SOIL_PIN A1
#define WATER_PIN A2

void setup() {
  Serial.begin(9600);
  randomSeed(analogRead(0));

  Serial.println("FloodSense System Starting...");
  delay(2000);
}

void loop() {

  int rain = analogRead(RAIN_PIN);
  int soil = analogRead(SOIL_PIN);
  int water = analogRead(WATER_PIN);

  // Fake data
  float temp = random(250, 350) / 10.0;
  float hum = random(400, 900) / 10.0;
  float pressure = random(9800, 10300) / 10.0;

  // Convert to usable values
  float rainfall_mm = map(rain, 0, 1023, 0, 300);
  float soil_pct = map(soil, 1023, 0, 0, 100);
  float water_m = map(water, 0, 1023, 0, 5);

  // ✅ SEND JSON (VERY IMPORTANT)
  Serial.print("DATA:{");
  Serial.print("\"rainfall_mm\":"); Serial.print(rainfall_mm);
  Serial.print(",\"temperature_c\":"); Serial.print(temp);
  Serial.print(",\"humidity_pct\":"); Serial.print(hum);
  Serial.print(",\"river_discharge_m3s\":10");
  Serial.print(",\"water_level_m\":"); Serial.print(water_m);
  Serial.print(",\"elevation_m\":300");
  Serial.print(",\"land_cover\":\"Agricultural\"");
  Serial.print(",\"soil_type\":\"Clay\"");
  Serial.print(",\"infrastructure\":1");
  Serial.print(",\"historical_floods\":1");
  Serial.print(",\"barometric_pressure_hpa\":"); Serial.print(pressure);
  Serial.print(",\"soil_moisture_pct\":"); Serial.print(soil_pct);
  Serial.println("}");

  delay(5000);
}