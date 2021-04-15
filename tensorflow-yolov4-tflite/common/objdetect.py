import os, base64
import tensorflow as tf
import core.utils as utils
import cv2
import numpy as np

def image_detect(saveImagePath, detectedImagePath, modelInfer, classesName, isOCR, **kwargs):

    INPUT_IMAGE_SIZE, IOU, SCORE, VIDEO_OUTPUT_FORMAT = kwargs.values()
    original_image = cv2.imread(saveImagePath)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    batch_data = tf.constant(image_data)
    pred_bbox = modelInfer(batch_data)

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

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    # pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    detectedImage = utils.draw_bbox_by_classes(original_image, pred_bbox, classesName=classesName, read_plate=isOCR)
    detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
    cv2.imwrite(detectedImagePath, detectedImage)
    
    try:
        with open(detectedImagePath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # print("encoded_string: ", encoded_string)
        img_url = f'data:image/jpg;base64,{encoded_string}'
    except Exception as e:
        print(e)
    return img_url


def gen_video_stream(filepath, detectedVideoPath, classesName, isOCR, **kwargs):
    """Video streaming generator function."""
    
    INPUT_IMAGE_SIZE, IOU, SCORE, VIDEO_OUTPUT_FORMAT = kwargs.values()
    vid = cv2.VideoCapture(filepath)
    
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
        pred_bbox = modelInfer(batch_data)
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

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        # pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        
        detectedImage = utils.draw_bbox_by_classes(frame, pred_bbox, classesName=classesName, read_plate=isOCR)
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


