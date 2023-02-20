#pip3 install ipython,pandas,opencv-python,numpy,torch torchvision torchaudio,psutil,tqdm,seaborn

import torch
import cv2
import os

from flask import Flask, request, render_template

app = Flask(__name__)

def detect_tumor(img_path):

  #load model
  model = torch.hub.load('ultralytics/yolov5', 'custom', path='tumor.pt', verbose=False)  # local model

  #read image
  img = cv2.imread(img_path)

  #convert image to rgb
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  #run model on image and save image to runs/detect/exp
  results = model(img)
  results.save()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():

  #gets user input image and saves it to static/uploads/*filename*
  file = request.files['file']
  filename = file.filename
  file_path = os.path.join('static', 'uploads', filename)
  file.save(file_path)

  #deletes any previous runs from rins/detect/exp/
  if len(os.listdir(os.path.join('runs','detect'))) == 2:
    os.remove(os.path.join('runs', 'detect', 'exp', "image0.jpg"))
    os.rmdir(os.path.join('runs', 'detect', 'exp'))
  if len(os.listdir(os.path.join('static', 'images'))) == 2:
     os.remove(os.path.join('static', 'images', 'image0.jpg'))
     os.rmdir(os.path.join('static', 'images', 'exp'))

  #runs tumor detection software
  detect_tumor(file_path)
  img = cv2.imread('./runs/detect/exp/image0.jpg')
  cv2.imwrite(os.path.join('static', 'images', 'image0.jpg'), img)
  return render_template('result.html', image_name='image0.jpg')

if __name__ == '__main__':
    app.run(debug=True)
