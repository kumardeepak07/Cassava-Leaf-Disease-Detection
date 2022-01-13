
from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('WebPage.html')


@app.route('/Cassava_Leaf_Disease_Detection', methods=['GET', 'POST'])
def classify_image():
    image_data = request.form['image_data']
    util.load_saved_artifacts()
    response = jsonify(util.classify_image(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(debug=True)

