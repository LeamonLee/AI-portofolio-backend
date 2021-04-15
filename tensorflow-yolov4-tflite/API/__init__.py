from flask import Flask
from API.config import Config
from flask_cors import CORS
import os
import logging

logger = logging.getLogger(__name__)
app = None

def initLogger():

    global logger

    dirpath = os.path.join(app.root_path, "LOG")

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    strToday = datetime.date.today()
    strToday = strToday.strftime("%Y-%m-%d")

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(name)s - %(lineno)s] %(message)s')
    file_handler = logging.FileHandler(f'{dirpath}/{strToday}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def create_app(config_class=Config):
    
    global app
    app = Flask(__name__, root_path=os.getcwd())
    print("os.getcwd(): ", os.getcwd(), "app.root_path: ", app.root_path)
    
    imageSavePath = os.path.join(app.root_path, "static", "upload", "images")
    videoSavePath = os.path.join(app.root_path, "static", "upload", "videos")
    print("imageSavePath: ", imageSavePath)
    print("videoSavePath: ", videoSavePath)

    if os.path.exists(imageSavePath) == False:
      os.makedirs(imageSavePath)
    if os.path.exists(videoSavePath) == False:
      os.makedirs(videoSavePath)

    app.config["IMAGE_UPLOADS"] = imageSavePath
    app.config["VIDEO_UPLOADS"] = videoSavePath
    app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
    app.config.from_pyfile("objdetect.cfg")

    dctODKwargs = {
      "INPUT_IMAGE_SIZE": app.config["INPUT_IMAGE_SIZE"],
      "IOU": app.config["IOU"], 
      "SCORE": app.config["SCORE"], 
      "VIDEO_OUTPUT_FORMAT": app.config["VIDEO_OUTPUT_FORMAT"]
    }

    app.config["OD_KWARGS"] = dctODKwargs

    print(app.config["INPUT_IMAGE_SIZE"])
    print(app.config["IOU"])
    print(app.config["SCORE"])
    print(app.config["VIDEO_OUTPUT_FORMAT"])

    CORS(app, supports_credentials=True)

    from API.main.routes import main
    from API.objdetect_lpr.routes import objdetect_lpr
    from API.objdetect_facemask.routes import objdetect_facemask
    from API.objdetect_coronavirus.routes import objdetect_coronavirus
    from API.imageclassification.routes import classification
    app.register_blueprint(main)
    app.register_blueprint(objdetect_lpr)
    app.register_blueprint(objdetect_facemask)
    app.register_blueprint(objdetect_coronavirus)
    app.register_blueprint(classification)

    return app