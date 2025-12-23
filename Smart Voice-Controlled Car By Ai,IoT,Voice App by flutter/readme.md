🚗🎙️ Smart Voice-Controlled Car using AI, IoT & Flutter
I’m excited to share my latest Big project: Smart Car Voice Control System, an end-to-end solution that allows controlling a robotic car using natural voice commands.
🔹 Project Overview
This system enables users to control a smart car by speaking commands in Arabic or English, which are processed using an AI-based NLP model and executed in real time through IoT communication.
🧠 AI & Backend
Trained an NLP Intent Classification model (Logistic Regression + TF-IDF)
Achieved ~93% accuracy
Built a Flask REST API for prediction and command handling
Mapped intents to low-level control commands (F, B, L, R, S)
Deployed the API publicly using ngrok
🌐 IoT & Hardware
ESP8266 (NodeMCU) for Wi-Fi & MQTT communication
HiveMQ Cloud (MQTT over TLS) for secure messaging
Arduino + Motor Driver (L298N) for motor control
Real-time execution of commands received from the server
📱 Mobile Application (Flutter)
Voice input using speech-to-text
Arabic language support (ar_SA)
Clean & modern UI with splash screen
Dynamic server configuration (HTTP / HTTPS)
Persistent settings using local storage
🔄 System Flow
Voice Command → Flutter App → Flask API → NLP Model → MQTT → ESP8266 → Motors
🛠️ Tech Stack
Python, Flask, scikit-learn
MQTT, HiveMQ, ESP8266
Arduino, Motor Driver
Flutter, Dart
NLP, Speech Recognition
🚀 Key Features
Multilingual voice commands (Arabic & English)
Real-time response
Secure cloud communication
Modular & scalable architecture
Ready for future enhancements (sensors, speed control, feedback, manual mode)
This project helped me gain strong hands-on experience in AI, IoT systems integration, backend development, and mobile applications.
