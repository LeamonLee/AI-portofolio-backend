FROM tensorflow/tensorflow:latest-gpu

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libgl1-mesa-dev \ 

RUN apt update && apt install -y \
    tesseract-ocr \
    libtesseract-dev

RUN pip install --upgrade pip && pip install setuptools

WORKDIR /usr/src/app

COPY . .

RUN pip install -r /usr/src/app/requirements-deploy.txt

# ENTRYPOINT [ "python3", "/usr/src/app/trainer.py" ]
CMD exec gunicorn --bind :5001 --workers 1 --threads 8 --timeout 0 run:app


