from flask import Flask, request, jsonify, render_template
import joblib
import re
import numpy as np
import paho.mqtt.client as mqtt
import ssl

app = Flask(__name__)

# -----------------------------
# Mapping intent -> car command
# -----------------------------
INTENT_TO_COMMAND = {
    "forward": "F",
    "backward": "B",
    "left": "L",
    "right": "R",
    "stop": "S",
}

# -----------------------------
# MQTT configuration (HiveMQ)
# -----------------------------
MQTT_BROKER = "da4f8ead70144159b7b192ae1a4b33d5.s1.eu.hivemq.cloud"  # غيّرها لعنوان الـ broker من HiveMQ
MQTT_PORT = 8883
MQTT_TOPIC = "car/control"
MQTT_USERNAME = "NLP_Car"  # غيّرها لاسم المستخدم من HiveMQ
MQTT_PASSWORD = "Abdallah2112004"  # غيّرها للباسورد من HiveMQ

mqtt_client = None
MQTT_CONNECTED = False


def init_mqtt():
    """الاتصال بـ MQTT مرة واحدة عند تشغيل السيرفر"""
    global mqtt_client, MQTT_CONNECTED
    try:
        client = mqtt.Client()
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
        client.connect(MQTT_BROKER, MQTT_PORT)
        mqtt_client = client
        MQTT_CONNECTED = True
        print("✅ Connected to MQTT broker")
    except Exception as e:
        MQTT_CONNECTED = False
        mqtt_client = None
        print(f"❌ MQTT connection failed: {e}")


def send_to_mqtt(command):
    """إرسال الأمر النهائي إلى MQTT لو الاتصال شغال"""
    if not MQTT_CONNECTED or mqtt_client is None:
        print("⚠️ MQTT not connected, cannot send command:", command)
        return
    try:
        mqtt_client.publish(MQTT_TOPIC, command)
        print("📡 Sent to MQTT:", command)
    except Exception as e:
        print("❌ Failed to publish to MQTT:", e)


# نحاول نعمل اتصال MQTT عند تشغيل التطبيق
init_mqtt()

# -----------------------------
# تحميل الموديل مرة واحدة مع معالجة الأخطاء
# -----------------------------
try:
    model = joblib.load("models/nlp_intent_model.joblib")
    MODEL_LOADED = True
except Exception as e:
    # لو في مشكلة في تحميل الموديل، نخليه None ونسجل الحالة
    model = None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)

# -----------------------------
# نفس clean_text اللي استخدمته في التدريب
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"(وقف|اوقف|توقف|ستوب)", "اقف", text)
    text = re.sub(r"قدامي", "قدام", text)
    text = re.sub(r"ورى", "ورا", text)
    return text

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    # نرسل حالة الموديل للـ GUI
    return render_template("index.html", model_loaded=MODEL_LOADED, result=None, error_msg=None)


def log_prediction(text, clean_cmd, intent):
    """طباعة أوتبت واضح في التيرمنال لكل طلب"""
    print("===================================")
    print("Input Text       :", text)
    print("Cleaned Text     :", clean_cmd)
    print("Predicted Intent :", intent)
    print("===================================")

@app.route("/predict", methods=["POST"])
def predict():
    # تأكيد أن الموديل متحمل قبل الاستخدام
    if not MODEL_LOADED or model is None:
        # HTML أو JSON حسب الطلب
        if request.is_json:
            return jsonify({"error": "Model not loaded on server"}), 500
        return render_template(
            "index.html",
            model_loaded=False,
            result=None,
            error_msg="لم يتم تحميل الموديل على السيرفر."
        ), 500

    # قراءة الداتا من JSON أو من فورم HTML
    text = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
    else:
        text = request.form.get("text", "")

    if not text:
        if request.is_json:
            return jsonify({"error": "No text provided"}), 400
        return render_template(
            "index.html",
            model_loaded=True,
            result=None,
            error_msg="الرجاء إدخال أمر نصي."
        ), 400

    clean_cmd = clean_text(text)
    intent = model.predict([clean_cmd])[0]
    command = INTENT_TO_COMMAND.get(intent, "S")
    confidence = None
    try:
        probs = model.predict_proba([clean_cmd])[0]
        confidence = float(np.max(probs) * 100)
    except Exception:
        confidence = None

    # طباعة واضحة في التيرمنال
    log_prediction(text, clean_cmd, intent)
    print("Final Command to Car:", command)
    # إرسال الأمر إلى MQTT
    send_to_mqtt(command)

    if request.is_json:
        return jsonify({
            "input": text,
            "clean_text": clean_cmd,
            "intent": intent,
            "command": command,
            "confidence": round(confidence, 2) if confidence is not None else None
        })

    # عرض النتيجة على الصفحة لو الطلب كان من فورم HTML
    result_data = {
        "input": text,
        "clean_text": clean_cmd,
        "intent": intent,
        "command": command,
        "confidence": round(confidence, 2) if confidence is not None else None
    }
    return render_template(
        "index.html",
        model_loaded=True,
        result=result_data,
        error_msg=None
    )

if __name__ == "__main__":
    app.run(debug=True)
