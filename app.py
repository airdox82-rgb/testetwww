import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
import threading
import uuid
import subprocess
import tempfile

status_lock = threading.Lock()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static')
CORS(app)  # Wichtig für deine GUI - muss vor den Routen definiert werden

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

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
    
    # Validiere Dateierweiterung
    allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    file_ext = os.path.splitext(f.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"Nicht unterstütztes Dateiformat: {file_ext}. Erlaubt: {', '.join(allowed_extensions)}"}), 400
    
    # Sichere Dateinamen
    filename = f.filename.replace(" ", "_")  # Ersetze Leerzeichen
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")  # Nur sichere Zeichen
    
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Prüfe ob Datei bereits existiert
    if os.path.exists(save_path):
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(save_path):
            filename = f"{base}_{counter}{ext}"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            counter += 1
    
    try:
        f.save(save_path)
        # Prüfe Dateigröße
        file_size = os.path.getsize(save_path)
        if file_size > 100 * 1024 * 1024:  # 100MB Limit
            os.remove(save_path)
            return jsonify({"error": "Datei zu groß (max. 100MB)"}), 400
            
        with status_lock:
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Sample hochgeladen: {filename} ({file_size} bytes)")
        return jsonify({"sample": filename})
        
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": f"Fehler beim Speichern der Datei: {str(e)}"}), 500

def _run_training():
    """Runs the actual training process."""
    with status_lock:
        status_data["progress"] = 0
        status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Training gestartet")

    try:
        # 1. ASR
        with status_lock:
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Starte Spracherkennung (ASR)")
        
        # Prüfe ob Upload-Ordner Dateien enthält
        if not os.listdir(UPLOAD_FOLDER):
            with status_lock:
                status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Fehler: Keine Dateien im Upload-Ordner gefunden")
                status_data["training"] = False
            return
        
        asr_output_dir = tempfile.mkdtemp()
        
        # Verwende Python-Executable aus dem aktuellen Environment
        python_executable = "python"
        
        asr_process = subprocess.run(
            [
                python_executable,
                "tools/asr/fasterwhisper_asr.py",
                "-i", UPLOAD_FOLDER,
                "-o", asr_output_dir,
                "-s", "large-v3",
                "-l", "de", # Assuming German
                "-p", "float16"
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 Minuten Timeout
        )
        
        if asr_process.returncode != 0:
            error_msg = asr_process.stderr or asr_process.stdout or "Unbekannter Fehler"
            with status_lock:
                status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] ASR fehlgeschlagen: {error_msg}")
                status_data["training"] = False
            return

        transcription_file = os.path.join(asr_output_dir, f"{os.path.basename(UPLOAD_FOLDER)}.list")
        
        # Prüfe ob Transkriptionsdatei erstellt wurde
        if not os.path.exists(transcription_file):
            with status_lock:
                status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Fehler: Transkriptionsdatei wurde nicht erstellt")
                status_data["training"] = False
            return
            
        with status_lock:
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] ASR abgeschlossen. Transkriptionsdatei: {transcription_file}")
            status_data["progress"] = 100 # For now, let's say ASR is the whole process

    except subprocess.TimeoutExpired:
        with status_lock:
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] ASR-Prozess hat das Zeitlimit überschritten")
            status_data["training"] = False
    except FileNotFoundError as e:
        with status_lock:
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] ASR-Skript nicht gefunden: {e}")
            status_data["training"] = False
    except Exception as e:
        with status_lock:
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Ein unerwarteter Fehler ist aufgetreten: {str(e)}")
            status_data["training"] = False
    finally:
        with status_lock:
            if status_data["training"]:  # Nur wenn noch aktiv
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
    
    try:
        # Validiere Eingabetext
        if not text or len(text.strip()) == 0:
            with status_lock:
                status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Fehler: Leerer Text für Synthese")
                status_data["synthesizing"] = False
            return
            
        if len(text) > 1000:  # Begrenze Textlänge
            with status_lock:
                status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Warnung: Text wurde auf 1000 Zeichen gekürzt")
            text = text[:1000]
        
        time.sleep(5) # Simulate work being done

        # Erzeuge eine eindeutige WAV-Datei
        output_filename = f"output_{uuid.uuid4().hex}.wav"
        dummy_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # Erstelle eine gültigere WAV-Datei (minimal aber korrekt)
        wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        
        with open(dummy_path, "wb") as f:
            f.write(wav_header)

        with status_lock:
            status_data["synthesizing"] = False
            status_data["output_file"] = output_filename
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Synthese beendet. Datei: {output_filename}")
            
    except Exception as e:
        with status_lock:
            status_data["log"].append(f"[{time.strftime('%H:%M:%S')}] Synthese-Fehler: {str(e)}")
            status_data["synthesizing"] = False

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