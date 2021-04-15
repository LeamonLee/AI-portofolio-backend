from flask import request, Blueprint, jsonify
from API import app, logger
from werkzeug.utils import secure_filename
import tensorflow as tf
import os, traceback
import numpy as np
import cv2

classification = Blueprint('classification', __name__, url_prefix='/classification')

food_classes = ['bread', 'dairy_product', 'dessert', 'egg', 'fried_food', 'meat', 
                'noodles_pasta', 'rice', 'seafood', 'soup', 'vegetable or fruit']

food_prediction_model = tf.keras.models.load_model('./checkpoints/tf-custom-food-recognition-416')

@classification.errorhandler(Exception)
def error_500(error):
    response = dict(message="500 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 500

@classification.errorhandler(KeyError)
def error_400(error):
    response = dict(message="400 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 400

def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]  
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@classification.route('/image_upload', methods=['POST'])
def image_upload():
  try:
    if request.files:
      image = request.files["image"]

      if image.filename == "":
          return jsonify({"message": "filename is empty, which is not allowed"}), 400

      if allowed_image(image.filename):
          filename = secure_filename(image.filename)
          image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
          return jsonify({"message": "The file is being uploaded successfully"})
      
      return jsonify({"message": "file extension should be either jpeg, jpg or png."}), 400
    
    return jsonify({"message": "please upload an image"}), 400
  except Exception as e:
    print("classification image_upload error: ", e)


@classification.route("/image_predict/<image_name>", methods=["POST"])
def image_predict(image_name):

  try:
    if request.files:
      image = request.files["image"]

      if image.filename == "":
          return jsonify({"message": "filename is empty, which is not allowed"}), 400

      if allowed_image(image.filename):
          filename = secure_filename(image.filename)
          image_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
          image.save(image_path)
      else:
        return jsonify({"message": "file extension should be either jpeg, jpg or png."}), 400
    else:
      return jsonify({"message": "please upload an image"}), 400

    image = cv2.imread(image_path) #BGR
    img = image.copy()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (229,229))
    image = image.astype("float32")
    image = image / 255.0
    np_image = np.expand_dims(image, axis=0) # (229,229,3) --> (1,229,229,3)

    predictions = food_prediction_model(np_image)
    predicted_class_idx = np.argmax(predictions) # [0.1, 0.5, 0.3] --> 1
    score = np.max(predictions)
    predicted_class = food_classes[predicted_class_idx]

    return jsonify({
      "predicted_class": predicted_class,
      "score": str(score)
    })
  except Exception as e:
    print("classification image_predict error: ", e)
