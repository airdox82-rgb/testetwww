from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Wichtig für deine GUI

status_data = {
    "training": False,
    "synthesizing": False,
    "progress": 0,
    "log": []
}

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(status_data)

@app.route("/api/samples", methods=["GET"])
def list_samples():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"samples": files})

@app.route("/api/upload_sample", methods=["POST"])
def upload_sample():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)
    status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Sample hochgeladen: {f.filename}")
    return jsonify({"sample": f.filename})

@app.route("/api/train", methods=["POST"])
def train():
    status_data["training"] = True
    status_data["progress"] = 0
    status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Training gestartet")

    # Simulierter Fortschritt
    for i in range(1, 11):
        time.sleep(0.5)
        status_data["progress"] = i * 10
    status_data["training"] = False
    status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Training beendet")
    return jsonify({"message": "Training abgeschlossen"})

@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "Kein Text angegeben"}), 400
    status_data["synthesizing"] = True
    status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Synthese gestartet: '{text}'")

    # Platzhalter: Erzeuge Dummy-WAV
    dummy_path = os.path.join(UPLOAD_FOLDER, "output.wav")
    with open(dummy_path, "wb") as f:
        f.write(b"RIFF....WAVEfmt ")  # Minimal WAV Header-Dummy

    status_data["synthesizing"] = False
    status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Synthese beendet")
    return send_file(dummy_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9871)
