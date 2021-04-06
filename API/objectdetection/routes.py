from flask import request, Blueprint, jsonify, Response, send_file
from API import app, logger
import io, time
import os, traceback

# =============================
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


objdetect = Blueprint('objdetect', __name__, url_prefix='/od')

@objdetect.errorhandler(Exception)
def error_500(error):
    
    response = dict(message="500 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 500


@objdetect.errorhandler(KeyError)
def error_400(error):
    
    response = dict(message="400 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 400


weightsPath = os.path.join(app.root_path, "checkpoints\yolov4-custom-lpr-416")
print("weightsPath: ", weightsPath)
INPUT_IMAGE_SIZE = 416
outputPath = os.path.join(app.root_path, "upload")
print("outputPath:", outputPath)
IOU = 0.45
SCORE = 0.25
VIDEO_OUTPUT_FORMAT = "XVID"

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

saved_model_loaded = tf.saved_model.load(weightsPath, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# API that returns JSON with classes found in images
@objdetect.route('/image_detect', methods=['POST'])
def image_detect():
    image = request.files["images"]
    imageFileName = image.filename
    saveImagePath = os.path.join(outputPath, imageFileName)
    image.save(saveImagePath)
    print("saveImagePath: ", saveImagePath)
    if imageFileName != "":
        original_image = cv2.imread(saveImagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU,
            score_threshold=SCORE
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        detectedImage = utils.draw_bbox(original_image, pred_bbox)  # type(detectedImage): <class 'numpy.ndarray'>
        detectedImage = Image.fromarray(detectedImage.astype(np.uint8)) # type(detectedImage): <class 'PIL.Image.Image'>
        detectedImage = cv2.cvtColor(np.array(detectedImage), cv2.COLOR_BGR2RGB)    # type(detectedImage3): <class 'numpy.ndarray'>
        
        detectedImageFileName = imageFileName.split('.')[0] + '_detected' + '.' + imageFileName.split('.')[1]
        detectedImagePath = os.path.join(outputPath, detectedImageFileName)
        print("detectedImagePath: ", detectedImagePath)
        cv2.imwrite(detectedImagePath, detectedImage)
        
        # frame = np.asarray(frame, np.uint8)
        # frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
        r, frame = cv2.imencode('.jpg', detectedImage)  # type(frame): <class 'numpy.ndarray'>
        print("type(frame):", type(frame))
        # return Response(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
        return send_file(detectedImagePath, mimetype='image/jpg')

    else:
        return jsonify({"response": "FileNotFoundError"}), 400

# API that returns JSON with classes found in images
@objdetect.route('/image_detect2', methods=['POST'])
def image_detect2():
    image = request.files["images"]
    imageFileName = image.filename
    saveImagePath = os.path.join(outputPath, imageFileName)
    image.save(saveImagePath)
    print("saveImagePath: ", saveImagePath)
    if imageFileName != "":
        original_image = cv2.imread(saveImagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU,
            score_threshold=SCORE
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        detectedImage = utils.draw_bbox(original_image, pred_bbox)
        detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
        detectedImageFileName = imageFileName.split('.')[0] + '_detected' + '.' + imageFileName.split('.')[1]
        detectedImagePath = os.path.join(outputPath, detectedImageFileName)
        print("detectedImagePath: ", detectedImagePath)
        cv2.imwrite(detectedImagePath, detectedImage)
        
        # frame = np.asarray(frame, np.uint8)
        # frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
        r, frame = cv2.imencode('.jpg', detectedImage)
        # return Response(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
        return send_file(detectedImagePath, mimetype='image/jpg')

    else:
        return jsonify({"response": "FileNotFoundError"}), 400

# API that returns JSON with classes found in images
@objdetect.route('/image_detect3', methods=['POST'])
def image_detect3():
    image = request.files["images"]
    imageFileName = image.filename
    saveImagePath = os.path.join(outputPath, imageFileName)
    image.save(saveImagePath)
    print("saveImagePath: ", saveImagePath)
    if imageFileName != "":
        original_image = cv2.imread(saveImagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU,
            score_threshold=SCORE
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        detectedImage = utils.draw_bbox(original_image, pred_bbox)
        detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
        detectedImageFileName = imageFileName.split('.')[0] + '_detected' + '.' + imageFileName.split('.')[1]
        detectedImagePath = os.path.join(outputPath, detectedImageFileName)
        print("detectedImagePath: ", detectedImagePath)
        cv2.imwrite(detectedImagePath, detectedImage)
        
        # 這個會失敗
        # detectedImage = Image.fromarray(detectedImage.astype(np.uint8))
        # return send_file(detectedImage, mimetype='image/jpg')
        
        r, frame = cv2.imencode('.jpg', detectedImage)
        return Response(frame.tobytes(), mimetype='image/jpg')

        # r, frame = cv2.imencode('.jpg', detectedImage)
        # return Response(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # return send_file(detectedImagePath, mimetype='image/jpg')

    else:
        return jsonify({"response": "FileNotFoundError"}), 400

@objdetect.route('/video_detect', methods=['POST'])
def video_detect():
    video = request.files["video"]
    videoFileName = video.filename
    if videoFileName != "":
        lstVideoFileName = videoFileName.split('.')
        saveVideoPath = os.path.join(outputPath, videoFileName)
        detectedVideoPath = lstVideoFileName[0] + '_detected' + '.' + lstVideoFileName[1]
        video.save(saveVideoPath)
        print("saveVideoPath: ", saveVideoPath)

        vid = cv2.VideoCapture(saveVideoPath)

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(VIDEO_OUTPUT_FORMAT)
        out = cv2.VideoWriter(detectedVideoPath, codec, fps, (width, height))

        frame_id = 0
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # image = Image.fromarray(frame)
            else:
                if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("Video processing complete")
                    break
                raise ValueError("No image! Try with another video format")
            
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            prev_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=IOU,
                score_threshold=SCORE
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            detectedImage = utils.draw_bbox(frame, pred_bbox)
            curr_time = time.time()
            exec_time = curr_time - prev_time
            # result = np.asarray(detectedImage)
            info = "time: %.2f ms" %(1000*exec_time)
            print(info)

            result = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)
            out.write(result)
            frame_id += 1

            r, frame = cv2.imencode('.jpg', result)
        return Response(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

    else:
        return jsonify({"response": "FileNotFoundError"}), 400