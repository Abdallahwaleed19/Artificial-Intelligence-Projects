// ===========================
// DOM Elements
// ===========================
const commandForm       = document.getElementById('commandForm');
const commandInput      = document.getElementById('commandInput');
const submitBtn         = document.getElementById('submitBtn');
const resultContainer   = document.getElementById('resultContainer');
const loadingContainer  = document.getElementById('loadingContainer');
const errorContainer    = document.getElementById('errorContainer');
const retryBtn          = document.getElementById('retryBtn');

const inputText         = document.getElementById('inputText');
const cleanText         = document.getElementById('cleanText');
const intentText        = document.getElementById('intentText');
const confidenceBar     = document.getElementById('confidenceBar');
const confidenceText    = document.getElementById('confidenceText');
const commandBadge      = document.getElementById('commandBadge');
const mqttStatusText    = document.getElementById('mqttStatusText');
const errorMessage      = document.getElementById('errorMessage');

// ===========================
// Helpers
// ===========================
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

// ===========================
// Fill quick command
// ===========================
function fillCmd(txt) {
    commandInput.value = txt;
    commandInput.focus();
}

// ===========================
// Form Submit
// ===========================
commandForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const cmd = commandInput.value.trim();
    if (!cmd) { showError('من فضلك أدخل أمر'); return; }

    showLoading();

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: cmd })
        });

        if (!res.ok) throw new Error('فشل الاتصال بالخادم');

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        displayResults(data);

    } catch (err) {
        showError(err.message || 'حدث خطأ أثناء معالجة الأمر');
    }
});

// ===========================
// Display Results
// ===========================
function displayResults(data) {
    hideLoading();
    hide(errorContainer);

    inputText.textContent  = data.input      || '—';
    cleanText.textContent  = data.clean_text || '—';
    intentText.textContent = data.intent     || '—';

    const conf = data.confidence || 0;
    confidenceBar.style.width  = conf + '%';
    confidenceText.textContent = conf + '%';

    // Confidence color
    confidenceBar.className = 'conf-fill';
    if (conf >= 80)      confidenceBar.classList.add('conf-fill--high');
    else if (conf >= 60) confidenceBar.classList.add('conf-fill--mid');
    else                 confidenceBar.classList.add('conf-fill--low');

    // Command badge
    if (commandBadge) {
        commandBadge.textContent = data.command || '—';
        commandBadge.className   = 'cmd-badge' + (data.command === 'STOP' ? ' cmd-badge--stop' : '');
    }

    // MQTT status
    if (mqttStatusText) {
        mqttStatusText.textContent = data.mqtt_sent ? '✓ MQTT DELIVERED' : '✗ MQTT FAILED';
        mqttStatusText.className   = 'mqtt-status ' + (data.mqtt_sent ? 'ok' : 'fail');
    }

    show(resultContainer);
}

// ===========================
// State Helpers
// ===========================
function showLoading() {
    hide(resultContainer);
    hide(errorContainer);
    show(loadingContainer);
    submitBtn.disabled = true;
}

function hideLoading() {
    hide(loadingContainer);
    submitBtn.disabled = false;
}

function showError(msg) {
    hideLoading();
    hide(resultContainer);
    errorMessage.textContent = msg;
    show(errorContainer);
}

// ===========================
// Retry
// ===========================
retryBtn.addEventListener('click', () => {
    hide(errorContainer);
    hide(resultContainer);
    commandInput.value = '';
    commandInput.focus();
});

// ===========================
// Keyboard Shortcuts
// ===========================
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        hide(resultContainer);
        hide(errorContainer);
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        commandForm.dispatchEvent(new Event('submit'));
    }
});

// ===========================
// Auto-focus
// ===========================
window.addEventListener('load', () => { commandInput.focus(); });
