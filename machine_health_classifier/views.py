from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
import matplotlib as plt
import cv2
import math
import base64
from keras.models import load_model
from keras.preprocessing import image

# Create your views here.

vgg16_model = load_model('./model_machine_health_pattern/models/vgg16_model.h5')
mobilenet_model = load_model('./model_machine_health_pattern/models/mobilenet_model.h5')
densenet_model = load_model('./model_machine_health_pattern/models/densenet_model.h5')
resnet_model = load_model('./model_machine_health_pattern/models/resnet_model.h5')
inception_model = load_model('./model_machine_health_pattern/models/inception_model.h5')
heatnet_model = load_model('./model_machine_health_pattern/models/heatnet_model.h5')

def index(request):
    context={'a':1}
    return render(request,'machine_health_classifier/index.html',context)

def result(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['file']
    fs=FileSystemStorage()
    filePath=fs.save(fileObj.name,fileObj)
    filePath=fs.url(filePath)

    userdir='.'+filePath
    userimg = image.load_img(userdir,target_size=(224,224))
    userimg_arr = image.img_to_array(userimg)  

    if request.method == "POST":
        if request.POST['dropdown'] == 'vgg16':
            userimg = tf.keras.applications.vgg16.preprocess_input(userimg_arr)
            userimg = np.array(userimg)
            userimg = userimg.reshape(1,224,224,3)
            result = vgg16_model.predict(x=userimg, verbose=0)
        elif request.POST['dropdown'] == 'mobilenet':
            userimg = tf.keras.applications.mobilenet.preprocess_input(userimg_arr)
            userimg = np.array(userimg)
            userimg = userimg.reshape(1,224,224,3)
            result = mobilenet_model.predict(x=userimg, verbose=0)
        elif request.POST['dropdown'] == 'densenet':
            userimg = tf.keras.applications.densenet.preprocess_input(userimg_arr)
            userimg = np.array(userimg)
            userimg = userimg.reshape(1,224,224,3)
            result = densenet_model.predict(x=userimg, verbose=0)
        elif request.POST['dropdown'] == 'resnet':
            userimg = tf.keras.applications.resnet50.preprocess_input(userimg_arr)
            userimg = np.array(userimg)
            userimg = userimg.reshape(1,224,224,3)
            result = resnet_model.predict(x=userimg, verbose=0)
        elif request.POST['dropdown'] == 'inception':
            userimg = tf.keras.applications.inception_v3.preprocess_input(userimg_arr)
            userimg = np.array(userimg)
            userimg = userimg.reshape(1,224,224,3)
            result = inception_model.predict(x=userimg, verbose=0)
        elif request.POST['dropdown'] == 'heatnet':
            userimg = cv2.imread(userdir)
            userimg = cv2.resize(userimg, (224, 224))
            userimg = cv2.applyColorMap(userimg, cv2.COLORMAP_HOT)
            userimg = image.img_to_array(userimg)
            userimg = np.array(userimg)
            userimg = userimg.reshape(1,224,224,3)
            result = heatnet_model.predict(x=userimg, verbose=0)


    if np.argmax(result)==0:
        label = "The machine's image is classified as Normal"
    else:
        label = "The machine's image is classified as Rusted"
    if result[0][0] > result[0][1]:
        predictedPercent = 'Prediction Amount ' + str(result[0][0]*100) + " %"
    else:
        predictedPercent = 'Prediction Amount ' + str(result[0][1]*100) + " %"
        

    context={'filePathName':filePath,'predictedLabel':label,'predictedPercent':predictedPercent}

    return render(request,'machine_health_classifier/index.html',context) 
