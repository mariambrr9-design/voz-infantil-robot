from flask import Flask, request, render_template_string, send_file
from TTS.api import TTS
import soundfile as sf, numpy as np, tempfile, re, os

app = Flask(__name__)
tts = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=False)

MAX_LEN = 250
def split_text(text):
    buf, out = "", []
    for o in re.split(r'(?<=[.!?]) +', text.strip()):
        if len(buf)+len(o) <= MAX_LEN: buf += o+" "
        else:
            if buf: out.append(buf.strip())
            buf = o+" "
    if buf: out.append(buf.strip())
    return out

def robot_light(wav, sr, f=0.25):
    t = np.arange(len(wav))/sr
    return wav * (0.5 + f*np.sin(2*np.pi*90*t))

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        texto   = request.form["texto"]
        emocion = request.form["emocion"]
        trozos  = split_text(texto)
        chunks  = [np.array(tts.tts(t, emotion=emocion, language="es")) for t in trozos]
        audio   = np.concatenate(chunks)
        sr      = tts.synthesizer.output_sample_rate
        audio   = robot_light(audio, sr)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sr)
            return send_file(tmp.name, as_attachment=True,
                             download_name="voz_infantil_robot.wav")
    return render_template_string(HTML)

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Voz Infantil Robot</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;margin:2rem;background:#111;color:#eee}
    textarea{width:100%;height:250px;background:#222;color:#0f0}
    select,button{font-size:1rem;margin-top:1rem}
  </style>
</head>
<body>
  <h1>🎙️ Voz Infantil Robot (Gratis)</h1>
  <form method="post">
    <label>Texto largo:</label><br/>
    <textarea name="texto" required></textarea><br/>
    <label>Emoción:</label>
    <select name="emocion">
      <option value="happy">Feliz</option>
      <option value="sad">Triste</option>
      <option value="angry">Enfadado</option>
      <option value="neutral">Neutral</option>
    </select><br/>
    <button type="submit">Crear audio</button>
  </form>
</body>
</html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
