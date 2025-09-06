import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
import threading
import uuid

status_lock = threading.Lock()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Wichtig für deine GUI

status_data = {
    "training": False,
    "synthesizing": False,
    "progress": 0,
    "log": [],
    "output_file": None
}

@app.route("/api/status", methods=["GET"])
def status():
    with status_lock:
        return jsonify(status_data)

@app.route("/api/samples", methods=["GET"])
def list_samples():
    # Filter out generated output files from the sample list
    files = [f for f in os.listdir(UPLOAD_FOLDER) if not f.startswith("output_")]
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
    with status_lock:
        status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Sample hochgeladen: {f.filename}")
    return jsonify({"sample": f.filename})

def _run_training():
    """Simulates a long-running training process in the background."""
    with status_lock:
        status_data["progress"] = 0
        status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Training gestartet")

    # Simulierter Fortschritt
    for i in range(1, 11):
        time.sleep(2)  # Simulate work being done
        with status_lock:
            status_data["progress"] = i * 10

    with status_lock:
        status_data["training"] = False
        status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Training beendet")

@app.route("/api/train", methods=["POST"])
def train():
    with status_lock:
        if status_data["training"] or status_data["synthesizing"]:
            return jsonify({"error": "Ein anderer Prozess läuft bereits"}), 409

        status_data["training"] = True
        status_data["output_file"] = None # Clear previous output file

    thread = threading.Thread(target=_run_training)
    thread.start()
    return jsonify({"message": "Training gestartet"})

def _run_synthesis(text):
    """Simulates a synthesis process in the background."""
    with status_lock:
        status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Synthese gestartet: '{text}'")
    
    time.sleep(5) # Simulate work being done

    # Erzeuge eine eindeutige WAV-Datei
    output_filename = f"output_{uuid.uuid4().hex}.wav"
    dummy_path = os.path.join(UPLOAD_FOLDER, output_filename)
    with open(dummy_path, "wb") as f:
        f.write(b"RIFF....WAVEfmt ")  # Minimal WAV Header-Dummy

    with status_lock:
        status_data["synthesizing"] = False
        status_data["output_file"] = output_filename
        status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Synthese beendet. Datei: {output_filename}")

@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "Kein Text angegeben"}), 400
    
    with status_lock:
        if status_data["training"] or status_data["synthesizing"]:
            return jsonify({"error": "Ein anderer Prozess läuft bereits"}), 409

        status_data["synthesizing"] = True
        status_data["output_file"] = None # Clear previous output file

    thread = threading.Thread(target=_run_synthesis, args=(text,))
    thread.start()
    return jsonify({"message": f"Synthese für '{text}' gestartet"})

@app.route("/api/output/<path:filename>")
def get_output(filename):
    """Serves the generated output file."""
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9871)
