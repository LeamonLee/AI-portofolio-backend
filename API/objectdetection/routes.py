# from flask import request, Blueprint, jsonify, Response
# from API import app, logger
# import logging

# # =============================
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
# import core.utils as utils
# from core.yolov4 import filter_boxes
# from tensorflow.python.saved_model import tag_constants
# from PIL import Image
# import cv2
# import numpy as np
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


# objdetect = Blueprint('objdetect', __name__, url_prefix='/od')

# @objdetect.errorhandler(Exception)
# def error_500(error):
    
#     response = dict(message="500 Error, {0}".format(error), detail=traceback.format_exc())
#     return jsonify(response), 500


# @objdetect.errorhandler(KeyError)
# def error_400(error):
    
#     response = dict(message="400 Error, {0}".format(error), detail=traceback.format_exc())
#     return jsonify(response), 400


# weightsPath = os.path.join(app.root_path + "/checkpoints/yolov4-custom-lpr-416")
# print("weightsPath: ", weightsPath)
# inputImageSize = 416
# outputPath = os.getcwd()
# IOU = 0.45
# SCORE = 0.25

# # config = ConfigProto()
# # config.gpu_options.allow_growth = True
# # session = InteractiveSession(config=config)
# # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

# saved_model_loaded = tf.saved_model.load(weightsPath, tags=[tag_constants.SERVING])
# infer = saved_model_loaded.signatures['serving_default']

# # API that returns JSON with classes found in images
# @flask_app.route('/detect', methods=['POST'])
# def human_detect():
#     image = request.files["images"]
#     imageFileName = image.filename
#     image.save(os.path.join(outputPath, imageFileName))
#     if imageFileName != "":
#         original_image = cv2.imread(f"./{imageFileName}")
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#         image_data = cv2.resize(original_image, (inputImageSize, inputImageSize))
#         image_data = image_data / 255.
#         image_data = image_data[np.newaxis, ...].astype(np.float32)
#         batch_data = tf.constant(image_data)
#         pred_bbox = infer(batch_data)

#         for key, value in pred_bbox.items():
#             boxes = value[:, :, 0:4]
#             pred_conf = value[:, :, 4:]

#         boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#             boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#             scores=tf.reshape(
#                 pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#             max_output_size_per_class=50,
#             max_total_size=50,
#             iou_threshold=IOU,
#             score_threshold=SCORE
#         )

#         pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
#         image = utils.draw_bbox(original_image, pred_bbox)
#         image = Image.fromarray(image.astype(np.uint8))

#         image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
#         cv2.imwrite(FLAGS.output, image)

#         # final_boxes = boxes.numpy()
#         # final_scores = scores.numpy()
#         # final_classes = classes.numpy()
#         # final_valid_detections = valid_detections.numpy
        
#         # array_boxes_detected = []
#         # if len(boxes)>0:
#         #     array_boxes_detected = get_human_box_detection(final_boxes,final_scores[0].tolist(),final_classes[0].tolist(),original_image.shape[0],original_image.shape[1])
#         # try:
#         #     return jsonify({"response":array_boxes_detected}), 200
#         # except FileNotFoundError:
#         #     abort(404)