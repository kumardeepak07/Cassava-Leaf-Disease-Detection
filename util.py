# -*- coding: utf-8 -*-

import json
import numpy as np
import base64
import cv2
import keras
import h5py
from PIL import Image

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):

    img = get_image_(file_path, image_base64_data)

    result = []
    res = []
    img = img.resize((456,456))
    img = np.expand_dims(img, axis=0)
    
    res.extend(__model.predict(img).argmax(axis = 1))
    prob = np.around(__model.predict(img)*100,2).tolist()[0]
    final = [ '%.2f' % elem for elem in prob ]
    

    result.append({
    'class': class_number_to_name(res[0]),
    'class_probability': final,
    'class_dictionary': __class_name_to_number
    })
    return result



def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    global __class_name_to_number
    global __class_number_to_name

    with open('class_dictionary.json') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}


    global __model
    if __model is None:
        with h5py.File('Cassava_model.h5', "r") as f:
            __model = keras.models.load_model(f)


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img = Image.fromarray(image, 'RGB')
    return img

    

def get_image_(image_path, image_base64_data):
    if image_path:
        img = Image.open(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
        
    return img


if __name__ == '__main__':
    load_saved_artifacts()







