from flask import Flask, jsonify, request
import json
import pickle
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['json'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


model = load_model('model.h5', compile=False)
max_length = 10
trunc_type = 'post'
padding_type = 'post'

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as label_handle:
    label_encoder = pickle.load(label_handle)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def extract_menu_names(json_data):
    items = []
    
    raw_json = json_data.get('raw_json', {})
    menu = raw_json.get('menu', [])

    if isinstance(menu, dict):
        items.append(menu.get('nm', ''))
    elif isinstance(menu, list):
        for item in menu:
            items.append(item.get('nm', ''))
    return items
 
def encode(texts):
    text_sequences = tokenizer.texts_to_sequences(texts)
    text_padded = pad_sequences(text_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return text_padded

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        json_file = request.files['json_file']
        if json_file and allowed_file(json_file.filename):
            filename = secure_filename(json_file.filename)
            json_file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(path, 'r') as file:
                json_data = json.load(file) 
            menu_names = extract_menu_names(json_data)
            padded_inputs = encode(menu_names)
            padded_inputs_tensor = tf.convert_to_tensor(np.array(padded_inputs))
            predictions = model.predict(padded_inputs_tensor)
            data = []
            for i, text in enumerate(menu_names):
                predicted_class = np.argmax(predictions[i])
                predicted_label = label_encoder.classes_[predicted_class]
                data.append([text, predicted_label])
            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "menu": [
                    {
                        "nm": data[i][0],
                        "category": data[i][1],
                    }
                    for i in range(len(data))
                ]
            }), 200
        else:
            print('in 400')
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid file format."
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == "__main__":
    app.run()