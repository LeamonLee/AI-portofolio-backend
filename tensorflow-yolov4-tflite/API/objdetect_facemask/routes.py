from flask import request, Blueprint, jsonify, Response
from API import app, logger
import time
import os, traceback, base64
from common.objdetect import *

import tensorflow as tf
from core.config import cfg
# import core.utils as utils
from tensorflow.python.saved_model import tag_constants
# import cv2
# import numpy as np
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

classesName = "facemask"
objdetect_facemask = Blueprint(f'objdetect_{classesName}', __name__, url_prefix=f'/od/{classesName}')

@objdetect_facemask.errorhandler(Exception)
def error_500(error):
    response = dict(message="500 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 500

@objdetect_facemask.errorhandler(KeyError)
def error_400(error):
    response = dict(message="400 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 400


# INPUT_IMAGE_SIZE = 416
# IOU = 0.45
# SCORE = 0.25
# # VIDEO_OUTPUT_FORMAT = "XVID"
# VIDEO_OUTPUT_FORMAT = 'MP4V'

# INPUT_IMAGE_SIZE = app.config["INPUT_IMAGE_SIZE"]
# IOU = app.config["IOU"]
# SCORE = app.config["SCORE"]
# VIDEO_OUTPUT_FORMAT = app.config["VIDEO_OUTPUT_FORMAT"]

dctODKwargs = app.config["OD_KWARGS"]

modelConfig = cfg.YOLO.MYCLASSES[classesName]
weightsPath = os.path.join(app.root_path, modelConfig["modelPath"])
print("weightsPath: ", weightsPath) 
saved_model_loaded = tf.saved_model.load(weightsPath, tags=[tag_constants.SERVING])
modelInfer = saved_model_loaded.signatures['serving_default']


# API that returns JSON with classes found in images
@objdetect_facemask.route('/image_detect', methods=['POST'])
def image_detect_route():

    image = request.files["images"]
    imageFileName = image.filename
    if imageFileName != "":
        saveImagePath = os.path.join(app.config["IMAGE_UPLOADS"], imageFileName)
        image.save(saveImagePath)
        print("saveImagePath: ", saveImagePath)

        detectedImageFileName = imageFileName.split('.')[0] + '_detected' + '.' + imageFileName.split('.')[1]
        detectedImagePath = os.path.join(app.config["IMAGE_UPLOADS"], detectedImageFileName)

        return image_detect(saveImagePath, detectedImagePath, modelInfer, classesName, **dctODKwargs)

    else:
        return jsonify({"response": "FileNotFoundError"}), 400

    # image = request.files["images"]
    # imageFileName = image.filename
    # saveImagePath = os.path.join(app.config["IMAGE_UPLOADS"], imageFileName)
    # image.save(saveImagePath)
    # print("saveImagePath: ", saveImagePath)
    # if imageFileName != "":
    #     original_image = cv2.imread(saveImagePath)
    #     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    #     image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    #     image_data = image_data / 255.
    #     image_data = image_data[np.newaxis, ...].astype(np.float32)
    #     batch_data = tf.constant(image_data)
    #     pred_bbox = modelInfer(batch_data)

    #     for key, value in pred_bbox.items():
    #         boxes = value[:, :, 0:4]
    #         pred_conf = value[:, :, 4:]

    #     boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    #         boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    #         scores=tf.reshape(
    #             pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    #         max_output_size_per_class=50,
    #         max_total_size=50,
    #         iou_threshold=IOU,
    #         score_threshold=SCORE
    #     )

    #     pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    #     detectedImage = utils.draw_bbox_by_classes(original_image, pred_bbox, classesName=classesName)
    #     detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
    #     detectedImageFileName = imageFileName.split('.')[0] + '_detected' + '.' + imageFileName.split('.')[1]
    #     detectedImagePath = os.path.join(app.config["IMAGE_UPLOADS"], detectedImageFileName)
    #     print("detectedImagePath: ", detectedImagePath)
    #     cv2.imwrite(detectedImagePath, detectedImage)
        
    #     try:
    #       with open(detectedImagePath, "rb") as image_file:
    #         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    #       # print("encoded_string: ", encoded_string)
    #       img_url = f'data:image/jpg;base64,{encoded_string}'
    #     except Exception as e:
    #       print(e)
    #     return img_url

    # else:
    #     return jsonify({"response": "FileNotFoundError"}), 400


@objdetect_facemask.route('/video_upload', methods=['POST'])
def video_upload():
    video = request.files["video"]
    videoFileName = video.filename
    if videoFileName != "":
        saveVideoPath = os.path.join(app.config["VIDEO_UPLOADS"], videoFileName)
        video.save(saveVideoPath)
        print("saveVideoPath: ", saveVideoPath)
        return jsonify({"response": "file uploaded successfully"}), 200
    else:
        return jsonify({"response": "FileNotFoundError"}), 400

# def gen_video_stream(filepath, filename):
#     """Video streaming generator function."""
#     vid = cv2.VideoCapture(filepath)
#     lstVideoFileName = filename.split('.')
#     detectedVideoFileName = lstVideoFileName[0] + '_detected' + '.' + lstVideoFileName[1]
#     detectedVideoPath = os.path.join(app.config["VIDEO_UPLOADS"], detectedVideoFileName)

#     # by default VideoCapture returns float instead of int
#     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(vid.get(cv2.CAP_PROP_FPS))
#     codec = cv2.VideoWriter_fourcc(*VIDEO_OUTPUT_FORMAT)
#     out = cv2.VideoWriter(detectedVideoPath, codec, fps, (width, height))

#     frame_id = 0
#     # Read until video is completed
#     while(vid.isOpened()):
#         # Capture frame-by-frame
#         return_value, frame = vid.read()
#         if return_value:
#             # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # image = Image.fromarray(frame)
#         else:
#             if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
#                 print("Video processing complete")
#                 break
#             raise ValueError("No image! Try with another video format")
        
#         frame_size = frame.shape[:2]
#         image_data = cv2.resize(frame, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
#         image_data = image_data / 255.
#         image_data = image_data[np.newaxis, ...].astype(np.float32)
#         prev_time = time.time()

#         batch_data = tf.constant(image_data)
#         pred_bbox = modelInfer(batch_data)
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
#         detectedImage = utils.draw_bbox_by_classes(frame, pred_bbox, classesName=classesName)
#         curr_time = time.time()
#         exec_time = curr_time - prev_time
#         # result = np.asarray(detectedImage)
#         info = "time: %.2f ms" %(1000*exec_time)
#         print(info)

#         result = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)
#         out.write(result)
#         frame_id += 1

#         r, frame = cv2.imencode('.jpg', result)
#         yield(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n')

@objdetect_facemask.route('/video_detect/<filename>', methods=['GET'])
def video_detect(filename):
    
    try:
        filepath = os.path.join(app.config["VIDEO_UPLOADS"], filename)
        if os.path.isfile(filepath):
            # return Response(gen_video_stream(filepath, filename, isOCR), mimetype='multipart/x-mixed-replace; boundary=frame')
            lstVideoFileName = filename.split('.')
            detectedVideoFileName = lstVideoFileName[0] + '_detected' + '.' + lstVideoFileName[1]
            detectedVideoPath = os.path.join(app.config["VIDEO_UPLOADS"], detectedVideoFileName)

            return Response(gen_video_stream(filepath, detectedVideoPath, classesName, **dctODKwargs), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return jsonify({"response": "FileNotFoundError"}), 400
    except Exception as e:
        print("video_detect error: ", e)
