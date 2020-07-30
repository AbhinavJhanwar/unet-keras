# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:09:36 2019

@author: abhinav.jhanwar
"""

from keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import load_model
from tqdm import tqdm

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img/255
        #print(img.shape, mask.shape)
        mask = mask/255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(batch_size,
                   train_path, 
                   image_folder,
                   mask_folder,
                   aug_dict,
                   image_color_mode = "grayscale",
                   mask_color_mode = "grayscale",
                   image_save_prefix  = "image",
                   mask_save_prefix  = "mask",
                   flag_multi_class = False,
                   num_class = 2, # only useful if flag_multi_class is set True
                   save_to_dir=None,
                   target_size=(256,256),
                   seed=67):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for
    image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder], 
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield(img, mask)


def validationGenerator(batch_size, 
                        val_path, 
                        image_folder, 
                        mask_folder, 
                        aug_dict, 
                        image_color_mode="grayscale",
                        mask_color_mode="grayscale", 
                        image_save_prefix ="image", 
                        mask_save_prefix ="mask",
                        flag_multi_class=False, 
                        num_class=2, 
                        save_to_dir=None, 
                        target_size=(256,256), 
                        seed=67):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    val_generator = zip(image_generator, mask_generator)
    for (img, mask) in val_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield(img, mask)


def testGenerator(test_path,num_image=200, target_size=(256,256), flag_multi_class=False, as_gray=True):
    dirs = os.listdir(test_path)
    for i in dirs:
        img = io.imread(os.path.join(test_path,i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

        
def makePredictions(save_path, test_data, model_path, custom_object):
    model = load_model(model_path, custom_objects=custom_object)
    test_images = os.listdir(test_data)
    for test_image in tqdm(test_images):
        x_test = load_img(os.path.join(test_data, test_image), target_size=(128,128))
        input_arr = img_to_array(x_test)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        prediction = model.predict(input_arr)
        prediction[prediction>0.5]=1
        prediction[prediction<=0.5]=0
        save_img(os.path.join(save_path, test_image.split('.')[0]+'_prediction.png'), prediction[0,:,:,:])
        