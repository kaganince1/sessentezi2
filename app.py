from flask import Flask, render_template, request
import inference

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sentez", methods = ["POST"])
def sentez():
    if request.method == "POST":
        metin = request.form["metin"]
        created_audio = inference.create_model(metin)
    return render_template("index.html", my_audio = created_audio)

if __name__ == "__main__":
    app.run();