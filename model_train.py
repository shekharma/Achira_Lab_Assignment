#upload data and mount to google drive to use google colab for training

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
import os

os.chdir('./drive/MyDrive/Achiralab/dataset/')

#Installation
pip install ultralytics


# yolo training 
!yolo detect train data=/content/drive/MyDrive/Achiralab/data.yaml model=yolov5n.pt epochs=5 imgsz=1024
!yolo detect train data=/content/drive/MyDrive/Achiralab/data.yaml model=yolov8n.pt epochs=5 imgsz=1024

# for more documentation
https://docs.ultralytics.com/
