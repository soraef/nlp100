from flask import Flask, render_template, request, jsonify
import MeCab
import os
import subprocess

def generate(text):
    parsed_text = mecab.parse(text)

    with open("92-input.txt", "w") as f:
        f.write(parsed_text)

    status = subprocess.call("onmt_translate -model demo-model_step_100000.pt -src 92-input.txt -output pred.txt -replace_unk -verbose > /dev/null 2>&1", shell=True)

    if status == 0:
        with open("pred.txt") as f:
            text = f.read()
            return text.replace(" ", "").rstrip("\n")

    return ""

app = Flask(__name__)

@app.route('/')
def chat():
    return render_template("chat.html", title="dialog server")

@app.route("/talk", methods=["POST"])
def talk():
    user_text = request.get_data().decode("utf-8")
    system_text = generate(user_text)
    response = jsonify({"text": system_text})
    return response, 201

if __name__ == "__main__":
    mecab = MeCab.Tagger("-Owakati")
    app.run(debug=True)