import os
import subprocess

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Voice-Controlled Car System Documentation</title>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0a2540;     /* Deep modern navy */
            --secondary: #00d4ff;   /* Vibrant cyan */
            --accent: #20c997;      /* Success/Accent green */
            --text-dark: #2d3748;
            --text-light: #718096;
            --bg-light: #f7fafc;
            --border: #e2e8f0;
        }

        @page {
            size: A4;
            margin: 0; /* Removing page margin entirely removes default browser headers/footers */
        }

        body {
            font-family: 'Inter', 'Cairo', sans-serif;
            line-height: 1.6;
            color: var(--text-dark);
            margin: 0;
            padding: 15mm; /* Restore safe margins for content */
            background-color: white;
            font-size: 10.5pt;
        }
        
        .avoid-break {
            page-break-inside: avoid;
        }

        /* --- Cover Page --- */
        .cover-page {
            position: relative;
            height: calc(29.7cm - 30mm); /* Exact A4 height minus padding */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            page-break-after: always;
            border: 2px solid var(--primary);
            padding: 1cm;
            box-sizing: border-box;
            background: linear-gradient(135deg, rgba(10,37,64,0.02) 0%, rgba(0,212,255,0.05) 100%);
            border-radius: 12px;
            margin-top: -5mm; /* adjust for body padding */
        }

        .uni-header {
            font-size: 18pt;
            font-weight: 800;
            color: var(--primary);
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .faculty-header {
            font-size: 14pt;
            font-weight: 500;
            color: var(--text-light);
            margin-bottom: 25px;
        }

        .cover-logo {
            width: 140px;
            height: auto;
            margin: 10px 0 25px 0;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }

        .project-title {
            font-size: 28pt;
            font-weight: 800;
            color: var(--primary);
            margin: 0 0 10px 0;
            line-height: 1.2;
            background: -webkit-linear-gradient(45deg, var(--primary), #1a5c9e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .project-subtitle {
            font-size: 16pt;
            font-weight: 400;
            color: var(--text-light);
            margin-bottom: 30px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }

        .course-info {
            background: white;
            padding: 15px 30px;
            border-radius: 12px;
            border-left: 5px solid var(--secondary);
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 30px;
            display: inline-block;
            text-align: left;
        }

        .course-info p {
            margin: 4px 0;
            font-size: 11pt;
        }

        .team-container {
            width: 100%;
            max-width: 550px;
            background: white;
            border-radius: 12px;
            padding: 15px 25px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.08);
            text-align: left;
            position: relative;
        }
        
        .team-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
        }

        .team-title {
            font-size: 13pt;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            text-align: center;
        }

        .team-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .team-list li {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid var(--border);
            font-size: 10.5pt;
        }

        .team-list li:last-child {
            border-bottom: none;
        }

        .team-list li .name {
            font-weight: 600;
            color: var(--primary);
        }

        .team-list li .id {
            font-family: 'Consolas', monospace;
            background: var(--bg-light);
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 10pt;
            color: var(--text-light);
            font-weight: 500;
        }

        /* --- Content Typography --- */
        h1, h2, h3, h4 {
            color: var(--primary);
            font-family: 'Inter', sans-serif;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
            line-height: 1.3;
        }

        h1 {
            font-size: 22pt;
            font-weight: 800;
            border-bottom: 2px solid var(--border);
            padding-bottom: 8px;
        }

        h2 {
            font-size: 16pt;
            font-weight: 700;
            display: flex;
            align-items: center;
            background-color: #f7fafc;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 4px solid var(--secondary);
        }

        h3 {
            font-size: 13pt;
            font-weight: 600;
            color: #1a365d;
        }

        p {
            margin-bottom: 1em;
            text-align: justify;
        }

        /* --- Lists --- */
        ul, ol {
            margin-bottom: 1em;
            padding-left: 2em;
        }

        li {
            margin-bottom: 0.5em;
        }
        
        li::marker {
            color: var(--secondary);
            font-weight: bold;
        }

        /* --- Alerts / Notes --- */
        .note {
            background: rgba(32, 201, 151, 0.1);
            border-left: 4px solid var(--accent);
            padding: 12px 15px;
            border-radius: 0 6px 6px 0;
            margin: 1em 0;
            font-weight: 500;
            color: #1b7352;
            display: flex;
            align-items: center;
        }

        .warning {
            background: #fffaf0;
            border-left: 4px solid #ed8936;
            padding: 12px 15px;
            border-radius: 0 6px 6px 0;
            margin: 1em 0;
            font-weight: 500;
            color: #c05621;
        }

        /* --- Tables --- */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            font-size: 10pt;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 0 0 1px var(--border);
        }

        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background-color: var(--bg-light);
            color: var(--primary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 9.5pt;
            letter-spacing: 0.5px;
        }

        tr:last-child td {
            border-bottom: none;
        }

        /* --- Code Blocks --- */
        .code-window {
            background: #1e1e1e;
            border-radius: 8px;
            margin: 1.5em 0;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }

        .code-header {
            background: #2d2d2d;
            padding: 8px 12px;
            display: flex;
            align-items: center;
        }

        .mac-dots {
            display: flex;
            gap: 5px;
        }

        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .dot.red { background: #ff5f56; }
        .dot.yellow { background: #ffbd2e; }
        .dot.green { background: #27c93f; }

        .code-title {
            margin-left: 12px;
            color: #858585;
            font-family: 'Consolas', monospace;
            font-size: 9pt;
        }

        pre {
            margin: 0;
            padding: 15px;
            overflow-x: auto;
            color: #d4d4d4;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 9.5pt;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        code {
            font-family: 'Consolas', 'Monaco', monospace;
            background-color: var(--bg-light);
            padding: 2px 4px;
            border-radius: 3px;
            color: #e53e3e;
            font-size: 0.9em;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            color: inherit;
        }
        
        /* Language specific coloring simulation */
        .keyword { color: #569cd6; }
        .string { color: #ce9178; }
        .comment { color: #6a9955; }
        .function { color: #dcdcaa; }

        /* Arabic text handling */
        .arabic {
            font-family: 'Cairo', sans-serif;
            direction: rtl;
            unicode-bidi: embed;
        }
        
        /* Two column layout for specific lists */
        .columns-2 {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 1.5em;
        }
        
        .columns-2 > div {
            flex: 1;
            min-width: 250px;
            background: #fdfdfd;
            padding: 12px 18px;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }

    </style>
</head>
<body>

    <!-- Cover Page -->
    <div class="cover-page">
        <div class="uni-header">Menoufia National University</div>
        <div class="faculty-header">Faculty of Computers &amp; Artificial Intelligence</div>
        
        <img src="C:/Users/AbdallahWaleed/Downloads/MY APPS/451515361_122102444084415918_4972823260009222385_n.jpg" alt="University Logo" class="cover-logo">
        
        <div class="project-title">Smart Voice-Controlled<br>Car System</div>
        <div class="project-subtitle">Project Documentation</div>
        
        <div class="course-info">
            <p><strong>Course:</strong> IoT Protocols</p>
            <p><strong>Instructor:</strong> Dr. Ahmed Badawey Rasas</p>
            <p><strong>Academic Year:</strong> 2026 - 2027</p>
        </div>
        
        <div class="team-container">
            <div class="team-title">Project Team</div>
            <ul class="team-list">
                <li><span class="name">1. Abdallah Waleed Kamal</span> <span class="id">20230192</span></li>
                <li><span class="name">2. Saif Sameh Fathy Elsawy</span> <span class="id">20230142</span></li>
                <li><span class="name">3. Abdelrahman Ali Ghonemi Ayad</span> <span class="id">20230174</span></li>
                <li><span class="name">4. Abdelrahman Fathy Awad Elbahrawey</span> <span class="id">20230176</span></li>
                <li><span class="name">5. Mohamed Garib Mohamed Abas</span> <span class="id">20230301</span></li>
                <li><span class="name">6. Mohamed Atef Mohamed Ali Shaheen</span> <span class="id">20230401</span></li>
            </ul>
        </div>
    </div>

    <!-- Content Flows Naturally -->
    <div>
        <h2>1. Introduction</h2>
        <p>This project presents a Smart Voice‑Controlled Car System that allows a robotic car to be controlled using Arabic and English voice commands. The system integrates Artificial Intelligence (Natural Language Processing) with IoT and Embedded Systems to convert human speech into real motor movements.</p>
        
        <div class="note">
            <span>✅ Flutter mobile application is excluded from this documentation as requested.</span>
        </div>

        <h2>2. Project Objective</h2>
        <p>The main goal of this project is to achieve seamless voice interaction with hardware. Specifically, the objectives are:</p>
        <ul>
            <li>Recognize Arabic and English voice commands with high precision.</li>
            <li>Classify the command intent utilizing a dedicated NLP model.</li>
            <li>Convert the detected intent into direct motor control signals.</li>
            <li>Control a physical robotic car remotely over a secure cloud MQTT connection.</li>
        </ul>

        <h2>3. System Architecture Overview</h2>
        <p>The system follows an End‑to‑End flow where data originates from the user and passes through an AI inference engine before reaching the hardware via IoT protocols:</p>
        
        <div class="avoid-break">
            <ol>
                <li><strong>Voice / Text Input</strong> (Captured via User Interface)</li>
                <li><strong>NLP Intent Classification Model</strong> (AI Inference)</li>
                <li><strong>Flask Backend API</strong> (HTTP Gateway)</li>
                <li><strong>Intent‑to‑Command Mapping</strong> (Logic Layer)</li>
                <li><strong>MQTT (HiveMQ Cloud)</strong> (Message Broker)</li>
                <li><strong>ESP8266</strong> (Wi‑Fi + MQTT Subscriber)</li>
                <li><strong>Arduino UNO</strong> (Motor Controller)</li>
                <li><strong>DC Motors</strong> (Physical Actuation)</li>
            </ol>
        </div>

        <h2>4. Technologies Used</h2>
        
        <div class="columns-2 avoid-break">
            <div>
                <h3>Software Technologies</h3>
                <ul>
                    <li>Python 3 &amp; Flask (REST API)</li>
                    <li>scikit‑learn &amp; joblib</li>
                    <li>Paho‑MQTT</li>
                    <li>SpeechRecognition</li>
                </ul>
            </div>
            
            <div>
                <h3>Cloud &amp; Hardware</h3>
                <ul>
                    <li>HiveMQ Cloud (Secure MQTT Broker)</li>
                    <li>ESP8266 (NodeMCU) &amp; Arduino UNO</li>
                    <li>L298N Motor Driver &amp; DC Motors</li>
                </ul>
            </div>
        </div>
    </div>

    <div>
        <h2>5. NLP Intent Classification Model</h2>
        
        <h3>5.1 Purpose of the Model</h3>
        <p>The NLP model acts as the brain of the system. It converts unstructured spoken or typed commands into predefined intents that explicitly represent car movements, effectively bridging the gap between natural language and machine commands.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Example Command (Arabic)</th>
                    <th>Extracted Intent</th>
                </tr>
            </thead>
            <tbody>
                <tr><td class="arabic">امشي قدام</td><td><code>forward</code></td></tr>
                <tr><td class="arabic">امشي ورا</td><td><code>backward</code></td></tr>
                <tr><td class="arabic">امشي يمين</td><td><code>right</code></td></tr>
                <tr><td class="arabic">امشي شمال</td><td><code>left</code></td></tr>
                <tr><td class="arabic">اقف</td><td><code>stop</code></td></tr>
            </tbody>
        </table>

        <h3>5.2 Dataset Structure</h3>
        <p>The dataset is stored in a structured CSV file ensuring a robust training process:</p>
        <table>
            <thead>
                <tr>
                    <th width="30%">Column</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr><td><strong>text</strong></td><td>Raw command text collected from various users to capture different dialects.</td></tr>
                <tr><td><strong>intent</strong></td><td>Target label for classification, mapped to the core movement states.</td></tr>
            </tbody>
        </table>

        <h3>5.3 Text Preprocessing Pipeline</h3>
        <p>To maximize accuracy and resilience to varied input formatting, rigorous preprocessing is applied:</p>
        <ul>
            <li><strong>Lowercasing:</strong> Standardizes English characters.</li>
            <li><strong>Whitespace Normalization:</strong> Removes trailing and extra inner spaces.</li>
            <li><strong>Arabic Character Normalization:</strong> Maps variations of Alif (أ, إ, آ) to a standard Alif (ا), Taa Marbuta (ة) to Haa (ه), and Alif Maqsura (ى) to Yaa (ي).</li>
            <li><strong>Synonym Mapping:</strong> Converts contextual variations (وقف, اوقف, توقف, ستوب) to a uniform command root (اقف).</li>
        </ul>
    </div>

    <div>
        <h3>5.4 NLP Model Code</h3>
        
        <div class="code-window">
            <div class="code-header">
                <div class="mac-dots">
                    <div class="dot red"></div>
                    <div class="dot yellow"></div>
                    <div class="dot green"></div>
                </div>
                <div class="code-title">train_nlp_model.py</div>
            </div>
<pre><code><span class="comment"># Intent Classification Model for Smart Car Control</span>
<span class="keyword">import</span> os, sys, re, joblib
<span class="keyword">import</span> pandas <span class="keyword">as</span> pd
<span class="keyword">import</span> numpy <span class="keyword">as</span> np
<span class="keyword">from</span> sklearn.model_selection <span class="keyword">import</span> train_test_split
<span class="keyword">from</span> sklearn.feature_extraction.text <span class="keyword">import</span> TfidfVectorizer
<span class="keyword">from</span> sklearn.linear_model <span class="keyword">import</span> LogisticRegression
<span class="keyword">from</span> sklearn.pipeline <span class="keyword">import</span> Pipeline

<span class="comment"># Load dataset and clean text</span>
df = pd.read_csv(<span class="string">'dataset.csv'</span>)
<span class="keyword">def</span> <span class="function">clean_text</span>(text):
    text = str(text).lower().strip()
    text = re.sub(<span class="string">r"[أإآ]"</span>, <span class="string">"ا"</span>, text)
    text = re.sub(<span class="string">r"ة"</span>, <span class="string">"ه"</span>, text)
    text = re.sub(<span class="string">r"ى"</span>, <span class="string">"ي"</span>, text)
    text = re.sub(<span class="string">r"(وقف|اوقف|توقف|ستوب)"</span>, <span class="string">"اقف"</span>, text)
    text = re.sub(<span class="string">r"قدامي"</span>, <span class="string">"قدام"</span>, text)
    text = re.sub(<span class="string">r"ورى"</span>, <span class="string">"ورا"</span>, text)
    <span class="keyword">return</span> text

df[<span class="string">'text_clean'</span>] = df[<span class="string">'text'</span>].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df[<span class="string">'text_clean'</span>], df[<span class="string">'intent'</span>], test_size=<span class="string">0.15</span>, random_state=<span class="string">42</span>, stratify=df[<span class="string">'intent'</span>]
)

<span class="comment"># Training pipeline</span>
pipeline = Pipeline([
    (<span class="string">'tfidf'</span>, TfidfVectorizer(ngram_range=(<span class="string">1</span>,<span class="string">3</span>))),
    (<span class="string">'clf'</span>, LogisticRegression(max_iter=<span class="string">2000</span>, C=<span class="string">10</span>, solver=<span class="string">'lbfgs'</span>))
])

pipeline.fit(X_train, y_train)

<span class="comment"># Save model</span>
os.makedirs(<span class="string">'models'</span>, exist_ok=<span class="keyword">True</span>)
joblib.dump(pipeline, <span class="string">'models/nlp_intent_model.joblib'</span>)</code></pre>
        </div>

        <h3>5.5 Model Performance</h3>
        <p>The final tuned model exhibits high reliability in intent mapping:</p>
        <ul>
            <li><strong>Accuracy achieved:</strong> ~93% on the validation set.</li>
            <li><strong>Robustness:</strong> Highly resilient for both formal Arabic and informal Egyptian commands.</li>
            <li><strong>Flexibility:</strong> Handles common spelling variations organically via n-gram TF-IDF vectorization.</li>
        </ul>
    </div>

    <div>
        <h2>6. Flask Backend API</h2>
        
        <h3>6.1 Backend Responsibilities</h3>
        <p>The Flask server acts as the central intelligence hub bridging the user interface and the hardware endpoint:</p>
        <ul>
            <li>Loads the pre-trained NLP model once into memory at startup for zero-latency inference.</li>
            <li>Receives command text payloads via a robust HTTP POST endpoint.</li>
            <li>Predicts the user's intent using the loaded model, then maps it to a single-character motor command.</li>
            <li>Publishes the resulting command to the HiveMQ MQTT broker.</li>
        </ul>

        <h3>6.2 Intent to Command Mapping</h3>
        <table>
            <thead>
                <tr>
                    <th>Classified Intent</th>
                    <th>Transmitted Motor Command (Payload)</th>
                </tr>
            </thead>
            <tbody>
                <tr><td><code>forward</code></td><td><strong>F</strong> (Drive Forward)</td></tr>
                <tr><td><code>backward</code></td><td><strong>B</strong> (Reverse)</td></tr>
                <tr><td><code>left</code></td><td><strong>L</strong> (Turn Left)</td></tr>
                <tr><td><code>right</code></td><td><strong>R</strong> (Turn Right)</td></tr>
                <tr><td><code>stop</code></td><td><strong>S</strong> (Halt Motors)</td></tr>
            </tbody>
        </table>

        <h3>6.3 Flask API Integration</h3>
        
        <div class="code-window">
            <div class="code-header">
                <div class="mac-dots">
                    <div class="dot red"></div>
                    <div class="dot yellow"></div>
                    <div class="dot green"></div>
                </div>
                <div class="code-title">app.py</div>
            </div>
<pre><code><span class="comment"># Flask API for Smart Car Control</span>
<span class="keyword">from</span> flask <span class="keyword">import</span> Flask, request, jsonify
<span class="keyword">import</span> joblib, re, ssl
<span class="keyword">import</span> paho.mqtt.client <span class="keyword">as</span> mqtt

app = Flask(__name__)

INTENT_TO_COMMAND = { <span class="string">"forward"</span>: <span class="string">"F"</span>, <span class="string">"backward"</span>: <span class="string">"B"</span>, <span class="string">"left"</span>: <span class="string">"L"</span>, <span class="string">"right"</span>: <span class="string">"R"</span>, <span class="string">"stop"</span>: <span class="string">"S"</span> }

<span class="comment"># Secure MQTT Setup</span>
client = mqtt.Client()
client.username_pw_set(<span class="string">"&lt;USERNAME&gt;"</span>, <span class="string">"&lt;PASSWORD&gt;"</span>)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
client.connect(<span class="string">"&lt;HiveMQ_URL&gt;"</span>, <span class="string">8883</span>)

model = joblib.load(<span class="string">'models/nlp_intent_model.joblib'</span>)

<span class="keyword">@app.route</span>(<span class="string">'/predict'</span>, methods=[<span class="string">'POST'</span>])
<span class="keyword">def</span> <span class="function">predict</span>():
    data = request.get_json()
    text = data.get(<span class="string">'text'</span>, <span class="string">''</span>)
    
    clean_cmd = clean_text(text) <span class="comment"># Custom cleaning func</span>
    intent = model.predict([clean_cmd])[<span class="string">0</span>]
    command = INTENT_TO_COMMAND.get(intent, <span class="string">'S'</span>)
    
    client.publish(<span class="string">"car/control"</span>, command) <span class="comment"># Dispatch to Broker</span>
    
    <span class="keyword">return</span> jsonify({ <span class="string">'input'</span>: text, <span class="string">'intent'</span>: intent, <span class="string">'command'</span>: command })</code></pre>
        </div>
    </div>

    <div>
        <h2>7. MQTT Communication (HiveMQ Cloud)</h2>
        <p>The IoT transport layer utilizes the MQTT protocol, known for lightweight and fast message delivery designed for constrained devices.</p>
        <ul>
            <li><strong>Security:</strong> All communication is encrypted via TLS over port 8883.</li>
            <li><strong>Routing &amp; Efficiency:</strong> The primary channel used is the <code>car/control</code> topic. The payload is optimized to a single ASCII character, minimizing bandwidth.</li>
        </ul>

        <h2>8. Hardware: ESP8266 &amp; Arduino Architecture</h2>
        
        <h3>8.1 ESP8266 &amp; Arduino Controllers</h3>
        <p>The ESP8266 connects to the local Wi-Fi, authenticates with HiveMQ, subscribes to <code>car/control</code>, and forwards received characters via Hardware Serial to the Arduino UNO, which then drives the L298N Motor Driver.</p>

        <div class="columns-2">
            <div>
                <h4>ESP8266 Snippet</h4>
<pre><code><span class="keyword">void</span> <span class="function">callback</span>(<span class="keyword">char</span>* topic, byte* payload, <span class="keyword">unsigned int</span> length) {
    <span class="comment">// Push raw command byte to Arduino</span>
    Serial.write((<span class="keyword">char</span>)payload[<span class="string">0</span>]);
}

<span class="keyword">void</span> <span class="function">loop</span>() {
    <span class="keyword">if</span> (!client.connected()) {
        client.connect(<span class="string">"ESP8266"</span>, <span class="string">"user"</span>, <span class="string">"pass"</span>);
        client.subscribe(<span class="string">"car/control"</span>);
    }
    client.loop();
}</code></pre>
            </div>
            <div>
                <h4>Arduino Snippet</h4>
<pre><code><span class="keyword">void</span> <span class="function">loop</span>() {
    <span class="keyword">if</span> (Serial.available()) {
        <span class="keyword">char</span> cmd = Serial.read();
        <span class="keyword">if</span> (cmd == <span class="string">'F'</span>) forward();
        <span class="keyword">else if</span> (cmd == <span class="string">'B'</span>) backward();
        <span class="keyword">else if</span> (cmd == <span class="string">'L'</span>) left();
        <span class="keyword">else if</span> (cmd == <span class="string">'R'</span>) right();
        <span class="keyword">else</span> stopMotors();
    }
}</code></pre>
            </div>
        </div>
    </div>

    <div>
        <h2>9. System Deployment &amp; Usage</h2>

        <h3>9.1 Hardware Assembly Sequence</h3>
        <ol>
            <li>Upload Arduino motor control code to the board.</li>
            <li>Connect DC motors to the L298N motor driver module.</li>
            <li>Connect ESP8266 TX/RX pins to Arduino RX/TX via level shifters/resistors.</li>
            <li>Power motors using a dedicated external power supply to prevent board resets.</li>
            <li>Verify MQTT connectivity via the serial monitor.</li>
        </ol>
        
        <div class="warning">
            <strong>Critical Deployment Note:</strong> The default backend server URL utilizes Ngrok (<code>https://ungraduating-kaylee-nakedly.ngrok-free.dev</code>). This is an ephemeral link and must be updated in the client application whenever the Flask server is restarted.
        </div>

        <h2>10. Mobile Client Context (Flutter)</h2>
        <p>While the Flutter application source code is excluded from this documentation, its role in the ecosystem is critical. The app provides the physical interface for the user, capturing voice streams via `ar_SA` configured speech-to-text engines.</p>
        
        <h3>Supported Voice Commands Catalog</h3>
        
        <div class="columns-2">
            <div>
                <h4>Movement Commands</h4>
                <ul class="arabic">
                    <li>"تقدم" / "امشي"</li>
                    <li>"تراجع" / "ارجع للخلف"</li>
                    <li>"يمين" / "اتجه يمين"</li>
                    <li>"يسار" / "اتجه يسار"</li>
                    <li>"قف" / "توقف"</li>
                </ul>
            </div>
            <div>
                <h4>Future Expansion (Speed)</h4>
                <ul class="arabic">
                    <li>"أسرع" / "زود السرعة"</li>
                    <li>"أبطأ" / "قلل السرعة"</li>
                </ul>
            </div>
        </div>

        <h2>11. Conclusion &amp; Results</h2>
        <p>The project successfully demonstrates a highly functional, real‑world application of Artificial Intelligence intersecting with IoT embedded systems. Testing indicates:</p>
        <ul>
            <li>The robotic vehicle responds instantly and accurately to varied Arabic and English voice commands.</li>
            <li>The NLP classification engine achieves a steady ~93% accuracy rate on out-of-sample data.</li>
            <li>The integration over the HiveMQ cloud ensures scalable, real-time wireless operation over the internet, allowing command transmission from anywhere in the world.</li>
        </ul>

        <h2>12. Future Improvements</h2>
        <ul>
            <li><strong>Manual Override Mode:</strong> Adding an on-screen joystick to the mobile client for traditional control.</li>
            <li><strong>Granular Speed Control:</strong> Utilizing PWM pins to allow voice commands like "drive slowly".</li>
            <li><strong>Obstacle Avoidance:</strong> Integrating Ultrasonic sensors to prevent collisions regardless of user intent.</li>
        </ul>
    </div>

</body>
</html>
"""

with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_path):
    print("Edge not found at default location.")
else:
    html_path = os.path.abspath("report.html")
    pdf_path = os.path.abspath("Professional_Smart_Car_Documentation.pdf")
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf=" + pdf_path,
        "file:///" + html_path.replace("\\", "/")
    ]
    subprocess.run(cmd, check=True)
    print("PDF generated successfully")
