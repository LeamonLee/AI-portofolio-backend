from flask import request, Blueprint, jsonify, Response, send_file
from API import app, logger
import io, time
import os, traceback, base64
from core.config import cfg

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

dctModel = {

}


INPUT_IMAGE_SIZE = 416
IOU = 0.45
SCORE = 0.25
# VIDEO_OUTPUT_FORMAT = "XVID"
VIDEO_OUTPUT_FORMAT = 'MP4V'

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

dctInfer = {}

for k,v in cfg.YOLO.MYCLASSES.items(): 
  weightsPath = os.path.join(app.root_path, v["modelPath"])
  print("weightsPath: ", weightsPath) 
  saved_model_loaded = tf.saved_model.load(weightsPath, tags=[tag_constants.SERVING])
  dctInfer[k] = saved_model_loaded.signatures['serving_default']

# API that returns JSON with classes found in images
@objdetect.route('/image_detect', methods=['POST'])
def image_detect1():
    image = request.files["images"]
    imageFileName = image.filename
    saveImagePath = os.path.join(app.config["IMAGE_UPLOADS"], imageFileName)
    image.save(saveImagePath)
    print("saveImagePath: ", saveImagePath)
    if imageFileName != "":
        original_image = cv2.imread(saveImagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = dctInfer(batch_data)

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
        detectedImagePath = os.path.join(app.config["IMAGE_UPLOADS"], detectedImageFileName)
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
    saveImagePath = os.path.join(app.config["IMAGE_UPLOADS"], imageFileName)
    image.save(saveImagePath)
    print("saveImagePath: ", saveImagePath)
    if imageFileName != "":
        original_image = cv2.imread(saveImagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = dctInfer(batch_data)

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
        detectedImagePath = os.path.join(app.config["IMAGE_UPLOADS"], detectedImageFileName)
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
    saveImagePath = os.path.join(app.config["IMAGE_UPLOADS"], imageFileName)
    image.save(saveImagePath)
    print("saveImagePath: ", saveImagePath)
    if imageFileName != "":
        original_image = cv2.imread(saveImagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = dctInfer(batch_data)

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
        detectedImagePath = os.path.join(app.config["IMAGE_UPLOADS"], detectedImageFileName)
        print("detectedImagePath: ", detectedImagePath)
        cv2.imwrite(detectedImagePath, detectedImage)
        
        # ???????????????
        # detectedImage = Image.fromarray(detectedImage.astype(np.uint8))
        # return send_file(detectedImage, mimetype='image/jpg')
        
        r, frame = cv2.imencode('.jpg', detectedImage)
        return Response(frame.tobytes(), mimetype='image/jpg')

        # r, frame = cv2.imencode('.jpg', detectedImage)
        # return Response(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # return send_file(detectedImagePath, mimetype='image/jpg')

    else:
        return jsonify({"response": "FileNotFoundError"}), 400


# API that returns JSON with classes found in images
@objdetect.route('/image_detect/<classesName>', methods=['POST'])
def image_detect(classesName):

    if classesName not in dctInfer.keys():
      return jsonify({"response": "classesName is not allowed"}), 400

    image = request.files["images"]
    imageFileName = image.filename
    saveImagePath = os.path.join(app.config["IMAGE_UPLOADS"], imageFileName)
    image.save(saveImagePath)
    print("saveImagePath: ", saveImagePath)
    if imageFileName != "":
        original_image = cv2.imread(saveImagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = dctInfer[classesName](batch_data)

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
        detectedImage = utils.draw_bbox_by_classes(original_image, pred_bbox, classesName=classesName)
        detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
        detectedImageFileName = imageFileName.split('.')[0] + '_detected' + '.' + imageFileName.split('.')[1]
        detectedImagePath = os.path.join(app.config["IMAGE_UPLOADS"], detectedImageFileName)
        print("detectedImagePath: ", detectedImagePath)
        cv2.imwrite(detectedImagePath, detectedImage)
        
        try:
          with open(detectedImagePath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
          print("encoded_string: ", encoded_string)
          img_url = f'data:image/jpg;base64,{encoded_string}'
        except Exception as e:
          print(e)
        return img_url

    else:
        return jsonify({"response": "FileNotFoundError"}), 400


@objdetect.route('/video_upload', methods=['POST'])
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

def gen_video_stream(filepath, filename):
    """Video streaming generator function."""
    vid = cv2.VideoCapture(filepath)
    lstVideoFileName = filename.split('.')
    detectedVideoFileName = lstVideoFileName[0] + '_detected' + '.' + lstVideoFileName[1]
    detectedVideoPath = os.path.join(app.config["VIDEO_UPLOADS"], detectedVideoFileName)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*VIDEO_OUTPUT_FORMAT)
    out = cv2.VideoWriter(detectedVideoPath, codec, fps, (width, height))

    frame_id = 0
    # Read until video is completed
    while(vid.isOpened()):
        # Capture frame-by-frame
        return_value, frame = vid.read()
        if return_value:
            # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
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
        pred_bbox = dctInfer(batch_data)
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
        yield(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n')

@objdetect.route('/video_detect/<classesName>/<filename>', methods=['GET'])
def video_detect(classesName, filename):
    
    if classesName not in dctInfer.keys():
      return jsonify({"response": "classesName is not allowed"}), 400
    
    try:
      filepath = os.path.join(app.config["VIDEO_UPLOADS"], filename)
      if os.path.isfile(filepath):
        return Response(gen_video_stream(filepath, filename), mimetype='multipart/x-mixed-replace; boundary=frame')
      else:
        return jsonify({"response": "FileNotFoundError"}), 400
    except Exception as e:
        print("video_detect error: ", e)

@objdetect.route('/video_detect2', methods=['POST'])
def video_detect2():
    video = request.files["video"]
    videoFileName = video.filename
    if videoFileName != "":
        lstVideoFileName = videoFileName.split('.')
        saveVideoPath = os.path.join(app.config["VIDEO_UPLOADS"], videoFileName)
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
            pred_bbox = dctInfer(batch_data)
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