from flask import Flask, request, jsonify, render_template
import joblib
import re
import numpy as np
import paho.mqtt.client as mqtt
import ssl
import logging
import threading
import time

app = Flask(__name__)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# Commands
# -----------------------------
INTENT_TO_COMMAND = {
    "forward":  "FORWARD",
    "backward": "BACKWARD",
    "left":     "LEFT",
    "right":    "RIGHT",
    "stop":     "STOP",
}

CONFIDENCE_THRESHOLD = 60.0

# -----------------------------
# MQTT
# -----------------------------
MQTT_BROKER   = "da4f8ead70144159b7b192ae1a4b33d5.s1.eu.hivemq.cloud"
MQTT_PORT     = 8883
MQTT_TOPIC    = "car/control"
MQTT_USERNAME = "NLP_Car"
MQTT_PASSWORD = "Abdallah2112004"

mqtt_client    = None
MQTT_CONNECTED = False
mqtt_lock      = threading.Lock()

mqtt_stats = {
    "sent":      0,
    "failed":    0,
    "last":      None,
    "last_time": None,
}


def on_connect(client, userdata, flags, rc):
    global MQTT_CONNECTED
    if rc == 0:
        MQTT_CONNECTED = True
        logging.info("✅ MQTT Connected")
    else:
        MQTT_CONNECTED = False
        RC_ERRORS = {
            1: "Wrong protocol",
            2: "Bad client ID",
            3: "Server unavailable",
            4: "Bad credentials",
            5: "Not authorized",
        }
        logging.error(f"❌ MQTT failed: {RC_ERRORS.get(rc, f'rc={rc}')}")


def on_disconnect(client, userdata, rc):
    global MQTT_CONNECTED
    MQTT_CONNECTED = False
    logging.warning(f"⚠️ MQTT Disconnected rc={rc}")


def on_publish(client, userdata, mid):
    logging.debug(f"✅ Delivered mid={mid}")


def init_mqtt():
    global mqtt_client, MQTT_CONNECTED

    if mqtt_client is not None:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        except Exception:
            pass

    try:
        client = mqtt.Client(
            client_id=f"Flask_NLP_{int(time.time())}",
            clean_session=True
        )
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        client.tls_set(
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS_CLIENT
        )

        client.on_connect    = on_connect
        client.on_disconnect = on_disconnect
        client.on_publish    = on_publish

        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()

        mqtt_client = client

        for _ in range(10):
            if MQTT_CONNECTED:
                break
            time.sleep(0.5)

        logging.info(f"🔄 MQTT init: {'connected' if MQTT_CONNECTED else 'pending'}")

    except Exception as e:
        MQTT_CONNECTED = False
        mqtt_client    = None
        logging.error(f"❌ MQTT init failed: {e}")


def send_to_mqtt(command: str) -> bool:
    global mqtt_client, MQTT_CONNECTED

    if not MQTT_CONNECTED or mqtt_client is None:
        logging.warning("⚠️ Not connected, reconnecting...")
        init_mqtt()

        if not MQTT_CONNECTED:
            mqtt_stats["failed"] += 1
            logging.error(f"❌ Still not connected, dropped: {command}")
            return False

    try:
        with mqtt_lock:
            result = mqtt_client.publish(
                MQTT_TOPIC,
                command,
                qos=1,
                retain=False
            )

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            mqtt_stats["sent"]      += 1
            mqtt_stats["last"]       = command
            mqtt_stats["last_time"]  = time.strftime("%H:%M:%S")
            logging.info(f"📡 MQTT ← [{command}]")
            return True
        else:
            mqtt_stats["failed"] += 1
            logging.error(f"❌ Publish rc={result.rc}")
            return False

    except Exception as e:
        mqtt_stats["failed"] += 1
        logging.error(f"❌ Publish error: {e}")
        return False


init_mqtt()


# -----------------------------
# NLP Model
# -----------------------------
MODEL_LOADED     = False
MODEL_LOAD_ERROR = None
model            = None

try:
    model        = joblib.load("models/nlp_intent_model.joblib")
    MODEL_LOADED = True
    logging.info("✅ NLP Model loaded")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    logging.error(f"❌ Model error: {e}")


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ة",       "ه", text)
    text = re.sub(r"ى",       "ي", text)

    text = re.sub(r"(وقف|اوقف|توقف|ستوب|قف|واقف)",                    "اقف",  text)
    text = re.sub(r"(قدامي|للامام|امام|للقدام|روح قدام|اتحرك قدام)",  "قدام", text)
    text = re.sub(r"(ورى|للخلف|الخلف|للورا|ارجع|روح ورا)",            "ورا",  text)
    text = re.sub(r"(يمين|اليمين|لليمين|دور يمين)",                    "يمين", text)
    text = re.sub(r"(شمال|اليسار|ليسار|للشمال|دور شمال)",              "شمال", text)

    return text


def predict_intent(clean_cmd: str):
    intent     = model.predict([clean_cmd])[0]
    confidence = None
    try:
        probs      = model.predict_proba([clean_cmd])[0]
        confidence = float(np.max(probs) * 100)
    except Exception:
        pass
    return intent, confidence


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        model_loaded=MODEL_LOADED,
        mqtt_connected=MQTT_CONNECTED,
        result=None,
        error_msg=None
    )


@app.route("/status")
def status():
    return jsonify({
        "model_loaded":   MODEL_LOADED,
        "mqtt_connected": MQTT_CONNECTED,
        "model_error":    MODEL_LOAD_ERROR,
        "threshold":      CONFIDENCE_THRESHOLD,
        "mqtt_stats":     mqtt_stats,
    })


@app.route("/test", methods=["POST"])
def test_command():
    data    = request.get_json(silent=True) or {}
    command = data.get("command", "").upper().strip()

    if command not in ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]:
        return jsonify({
            "error": "Invalid command",
            "valid": ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]
        }), 400

    ok = send_to_mqtt(command)
    return jsonify({
        "command": command,
        "sent":    ok,
        "stats":   mqtt_stats,
    })


@app.route("/predict", methods=["POST"])
def predict():

    if not MODEL_LOADED or model is None:
        err = {"error": "Model not loaded", "detail": MODEL_LOAD_ERROR}
        if request.is_json:
            return jsonify(err), 500
        return render_template(
            "index.html", model_loaded=False,
            mqtt_connected=MQTT_CONNECTED,
            result=None, error_msg="الموديل غير متاح"
        ), 500

    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
    else:
        text = request.form.get("text", "").strip()

    if not text:
        if request.is_json:
            return jsonify({"error": "No text provided"}), 400
        return render_template(
            "index.html", model_loaded=True,
            mqtt_connected=MQTT_CONNECTED,
            result=None, error_msg="من فضلك أدخل أمر"
        ), 400

    clean_cmd = clean_text(text)

    try:
        intent, confidence = predict_intent(clean_cmd)
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

    if intent not in INTENT_TO_COMMAND:
        logging.warning(f"⚠️ Unknown intent '{intent}' → STOP")
        intent = "stop"

    low_confidence = False
    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        logging.warning(f"⚠️ Low conf {confidence:.1f}% → STOP")
        intent         = "stop"
        low_confidence = True

    command = INTENT_TO_COMMAND[intent]

    logging.info(
        f"'{text}' → '{clean_cmd}' | {intent} | {command} "
        f"| {f'{confidence:.1f}%' if confidence else 'N/A'}"
    )

    mqtt_ok = send_to_mqtt(command)

    result_data = {
        "input":          text,
        "clean_text":     clean_cmd,
        "intent":         intent,
        "command":        command,
        "confidence":     round(confidence, 2) if confidence else None,
        "low_confidence": low_confidence,
        "mqtt_sent":      mqtt_ok,
    }

    if request.is_json:
        return jsonify(result_data)

    return render_template(
        "index.html",
        model_loaded=True,
        mqtt_connected=MQTT_CONNECTED,
        result=result_data,
        error_msg=None
    )


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
