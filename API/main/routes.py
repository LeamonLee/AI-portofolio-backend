from flask import request, Blueprint, jsonify, make_response, Response, send_file
from API import app, logger
import os,traceback
import cv2
import io, base64

main = Blueprint('main', __name__, url_prefix='/')

@main.errorhandler(Exception)
def error_500(error):
    
    response = dict(message="500 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 500


@main.errorhandler(KeyError)
def error_400(error):
    
    response = dict(message="400 Error, {0}".format(error), detail=traceback.format_exc())
    return jsonify(response), 400


@main.route("/")
@main.route("/home")
def home():
    logger.debug("Received home request...")
    return "Welcome Object Detection API System!"


def webcam_process():
  cap = cv2.VideoCapture(0)

  # Read until video is completed
  fps=0
  st=0
  frames_to_count=20
  cnt=0

  while(cap.isOpened()):
    ret, img = cap.read()

    if ret == True:
      if cnt == frames_to_count:
        try: # To avoid divide by 0 we put it in try except

          fps = round(frames_to_count/(time.time()-st))
          st = time.time()
          cnt=0
        except:
          pass

      cnt+=1
      frame = cv2.imencode('.JPEG', img,[cv2.IMWRITE_JPEG_QUALITY,20])[1].tobytes()
      # time.sleep(0.016)

      # ret, jpeg = cv2.imencode('.jpg', image)
      # frame = jpeg.tobytes()
      # yield (b'--frame\r\n'
      #          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    else:
      print("Webcam is closed!")
      break

@main.route("/webcam")
def webcam_feed():
	
	return Response(webcam_process(),mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route("/image")
def image_feed():

  imagePath = os.path.join(app.root_path, "data/kite.jpg")

  # Method1
  # with open(imagePath, "rb") as image_file:
  #   image_binary = io.BytesIO(image_file.read())
  # return send_file(image_binary, mimetype='image/jpg')
  
  # Method2
  # return send_file(imagePath, mimetype='image/jpg')

  # Method3
  # img = cv2.imread(imagePath)
  # frame = cv2.imencode('.JPEG', img,[cv2.IMWRITE_JPEG_QUALITY,20])[1].tobytes()
  # imgResponse = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
  # return Response(imgResponse, mimetype='multipart/x-mixed-replace; boundary=frame')
  # return Response(imgResponse, mimetype='multipart/x-mixed-replace')  # Doesn't work

  # Method4 不能用make response回傳image, 要用Response
  # with open(imagePath, "rb") as image_file:
  #   image_binary = io.BytesIO(image_file.read())
  # response = make_response(image_binary)
  # response.headers.set('Content-Type', 'image/jpeg')
  # response.headers.set(
  #     'Content-Disposition', 'attachment', filename='download.jpg')
  # return response

  # Method4
  # with open(imagePath, "rb") as image_file:
  #   image_binary = io.BytesIO(image_file.read())
  # return Response(image_binary, mimetype='image/jpg')
  
  # 不能用base64
  # https://stackoverflow.com/questions/35828806/use-base64-string-from-url-in-src-tag-of-image
  # Method5 base64
  # with open(imagePath, "rb") as image_file:
  #   encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
  # img_url = f'data:image/jpg;base64,{encoded_string}'
  # image_binary = img_url.encode("utf-8")
  # return img_url    # base64不能直接這樣回傳
  # return Response(image_binary)


  # with open(imagePath, "rb") as image_file:
  #   encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
  # return encoded_string
  # return Response(encoded_string)


  # with open(imagePath, "rb") as image_file:
  #   encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
  # uri = ("data:" + 
  #      request.headers['Content-Type'] + ";" +
  #      "base64," + encoded_string)
  # return uri