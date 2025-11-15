FROM python:3.10

WORKDIR /analoguegaugereader
COPY . /analoguegaugereader

RUN pip install -r requirements.txt 

CMD ["python3","src/circle_detection.py"]