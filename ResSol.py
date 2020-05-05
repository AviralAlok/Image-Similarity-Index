# 1. import the necessary packages
!pip install tensorflow==1.12.0
import tensorflow as tf
import math
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np

# 2. Getting the resnet Model
def predict(img_path : str, model: Model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

# 3. Finding the Euclidean Distance using matrix Vector norm
def Difference(f1, f2):
    return np.linalg.norm(f1-f2)

# 4. Function to load images
def func():
    feature_vectors: dict = {}
    model = ResNet50(weights='imagenet')
    feature_vectors["img1_path"]=predict("img1_path",model)[0]
    feature_vectors["img2_path"]=predict("img2_path",model)[0]
    x=Difference(feature_vectors["img1_path"],feature_vectors["img2_path"])
    print (math.exp(-(x**2)/(2*0.74*0.74)))

# 5. Calling function and printing m
func()