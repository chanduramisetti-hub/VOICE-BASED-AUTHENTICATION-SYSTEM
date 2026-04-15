import os
import numpy as np
import soundfile as sf
import subprocess
import torch

from flask import Flask, request, render_template, jsonify
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

app = Flask(__name__)

DB_FOLDER = "voice_db"
os.makedirs(DB_FOLDER, exist_ok=True)

# Load pretrained speaker verification model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models"
)

THRESHOLD = 0.65


# ---------------- AUDIO CONVERSION ---------------- #

def convert_audio(input_file, output_file):
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", input_file,
            "-ar", "16000",
            "-ac", "1",
            output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("Audio conversion error:", e)


# ---------------- EMBEDDING ---------------- #

def get_embedding(audio_path):

    signal, fs = sf.read(audio_path)

    signal = torch.tensor(signal).float()

    if len(signal.shape) > 1:
        signal = signal.mean(dim=1)

    # normalize audio
    signal = signal / torch.max(torch.abs(signal))

    signal = signal.unsqueeze(0)

    emb = verification.encode_batch(signal)

    emb = emb.squeeze().detach().numpy()

    return emb


# ---------------- PAGE ROUTES ---------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register_page")
def register_page():
    return render_template("register.html")


@app.route("/login_page")
def login_page():
    return render_template("login.html")


# ---------------- REGISTER ---------------- #

@app.route("/register", methods=["POST"])
def register():

    try:

        username = request.form.get("username")

        if not username:
            return jsonify({"status": "error", "message": "Username required"})

        audio = request.files.get("audio")

        if audio is None:
            return jsonify({"status": "error", "message": "Audio not received"})

        webm_file = "temp_register.webm"
        wav_file = "temp_register.wav"

        audio.save(webm_file)

        convert_audio(webm_file, wav_file)

        embedding = get_embedding(wav_file)

        os.makedirs(DB_FOLDER, exist_ok=True)

        np.save(os.path.join(DB_FOLDER, username + ".npy"), embedding)

        if os.path.exists(webm_file):
            os.remove(webm_file)

        if os.path.exists(wav_file):
            os.remove(wav_file)

        return jsonify({
            "status": "success",
            "message": "Voice Registered Successfully"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# ---------------- LOGIN ---------------- #

@app.route("/login", methods=["POST"])
def login():

    try:

        username = request.form.get("username")

        if not username:
            return jsonify({"status": "error", "message": "Username required"})

        audio = request.files.get("audio")

        if audio is None:
            return jsonify({"status": "error", "message": "Audio not received"})

        stored_path = os.path.join(DB_FOLDER, username + ".npy")

        if not os.path.exists(stored_path):

            return jsonify({
                "status": "error",
                "message": "User not registered"
            })

        webm_file = "temp_login.webm"
        wav_file = "temp_login.wav"

        audio.save(webm_file)

        convert_audio(webm_file, wav_file)

        new_embedding = get_embedding(wav_file)
        stored_embedding = np.load(stored_path)

        similarity = 1 - cosine(new_embedding, stored_embedding)

        if os.path.exists(webm_file):
            os.remove(webm_file)

        if os.path.exists(wav_file):
            os.remove(wav_file)

        if similarity > THRESHOLD:

            return jsonify({
                "status": "success",
                "message": "Login Successful",
                "similarity": float(similarity)
            })

        else:

            return jsonify({
                "status": "failed",
                "message": "Voice Not Matched",
                "similarity": float(similarity)
            })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# ---------------- RUN SERVER ---------------- #

if __name__ == "__main__":
    app.run(debug=True)