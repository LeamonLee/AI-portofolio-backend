

def image_detect(image, saveImagePath, detectedImagePath, modelInfer, **kwargs):

    imageFileName = image.filename
    image.save(saveImagePath)
    
    INPUT_IMAGE_SIZE, IOU, SCORE, VIDEO_OUTPUT_FORMAT = kwargs.values()

    if imageFileName != "":
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

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        detectedImage = utils.draw_bbox_by_classes(original_image, pred_bbox, classesName=classesName)
        detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
        cv2.imwrite(detectedImagePath, detectedImage)
        
        try:
          with open(detectedImagePath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
          # print("encoded_string: ", encoded_string)
          img_url = f'data:image/jpg;base64,{encoded_string}'
        except Exception as e:
          print(e)
        return 200, img_url

    else:
        return 400, {"response": "FileNotFoundError"}

def gen_video_stream(filepath, filename, detectedVideoPath):
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

@objdetect_facemask.route('/video_detect/<filename>', methods=['GET'])
def video_detect(filename):
    
    try:
      filepath = os.path.join(app.config["VIDEO_UPLOADS"], filename)
      if os.path.isfile(filepath):
        return Response(gen_video_stream(filepath, filename), mimetype='multipart/x-mixed-replace; boundary=frame')
      else:
        return jsonify({"response": "FileNotFoundError"}), 400
    except Exception as e:
        print("video_detect error: ", e)
