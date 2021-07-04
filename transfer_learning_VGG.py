'''
Descripttion: 
version: 
Author: LiQiang
Date: 2021-07-03 22:02:16
LastEditTime: 2021-07-04 08:55:56
'''
import os
from keras import models
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten,Dense,Dropout,Input
os.environ['CUDA_VISIBLE_DEVICES']='0'

def vgg():
    # load model
    base_model=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(64,64,3)))
    # summary=base_model.summary()
    # print(summary)
    # base_model.trainable=False
    # freeze some Conv layers
    # base_model.get_layer('block1_conv1').trainable=False
    # base_model.get_layer('block1_conv2').trainable=False
    # base_model.get_layer('block2_conv1').trainable=False
    # base_model.get_layer('block2_conv2').trainable=False

    # add new classifier layers
    flat1=Flatten(name="flatten")(base_model.output)
    class1=Dense(512,activation='relu')(flat1)
    output=Dense(3,activation='softmax')(class1)

    # define new model
    model=Model(inputs=base_model.inputs,outputs=output)
    for layer in base_model.layers:
        layer.trainable=False
    model.summary()
    return model


if __name__ =='__main__':
    print(vgg())