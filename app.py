import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flask import Flask, render_template, request

from PIL import Image



app = Flask(__name__)


model = load_model('vgg_model.h5')




@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')










def get_prediction(path):
    image_width, image_height = 150, 150
    img = image.load_img(path, target_size=(image_width, image_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    prediction = model.predict_classes(img)
    if prediction[0][0] == 0:
        return 'Hot Dog'
    else:
        return 'Not Hot Dog'



@app.route("/predict", methods=["POST"])
def prediction_service():
	file = request.files['input_image']
	im = Image.open(file)
	imagesrc = f'static/{file.filename}'
	im.save(imagesrc, 'PNG')
	pred = get_prediction(imagesrc)
	return render_template("index.html", pred=pred, imagesrc=imagesrc) 








app.run(debug=True, host='0.0.0.0')
