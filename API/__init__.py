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
    
    CORS(app, supports_credentials=True)

    from API.main.routes import main
    app.register_blueprint(main)

    return app