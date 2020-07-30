# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:09:36 2019

@author: abhinav.jhanwar
"""

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from evaluate import my_iou_metric

def unet(input_size=(256,256,1), verbose=True, pretrained_weights=None):
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2, 2))(drop4)

    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, (2, 2), padding = 'same', strides = (2, 2))(drop5)
    merge6 = concatenate([up6, drop4])
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), padding = 'same', strides = (2, 2))(conv6)
    merge7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), padding = 'same', strides = (2, 2))(conv7)
    merge8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), padding = 'same', strides = (2, 2))(conv8)
    merge9 = concatenate([up9, conv1])
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', my_iou_metric])
    
    if verbose:
        print(model.summary())

    if(pretrained_weights):
        print('Using pretrained weights:', pretrained_weights)
        model.load_weights(pretrained_weights)
        
    return model


