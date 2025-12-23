#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// --------------------
// WiFi credentials
// --------------------
const char* ssid = "Abdallah";
const char* password = "abdallah2112004";

// --------------------
// MQTT (HiveMQ Cloud)
// --------------------
const char* mqtt_server = "da4f8ead70144159b7b192ae1a4b33d5.s1.eu.hivemq.cloud";
const int mqtt_port = 8883;
const char* mqtt_user = "NLP_Car";
const char* mqtt_pass = "Abdallah2112004";
const char* mqtt_topic = "car/control";

// --------------------
// Objects
// --------------------
WiFiClientSecure espClient;
PubSubClient client(espClient);

// --------------------
// MQTT Callback
// --------------------
void callback(char* topic, byte* payload, unsigned int length) {
  if (length == 0) return;

  char command = (char)payload[0];   // F / B / L / R / S

  Serial.print("📩 MQTT Command: ");
  Serial.println(command);

  // إرسال الأمر للـ Arduino
  Serial.write(command);
}

// --------------------
// Connect to WiFi
// --------------------
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n✅ WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// --------------------
// Connect to MQTT
// --------------------
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection... ");

    if (client.connect("ESP8266_Client", mqtt_user, mqtt_pass)) {
      Serial.println("✅ connected");
      client.subscribe(mqtt_topic);
      Serial.print("Subscribed to: ");
      Serial.println(mqtt_topic);
    } else {
      Serial.print("❌ failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

// --------------------
// Setup
// --------------------
void setup() {
  Serial.begin(9600);

  // إعداد TLS بدون شهادة (HiveMQ)
  espClient.setInsecure();

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

// --------------------
// Loop
// --------------------
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
