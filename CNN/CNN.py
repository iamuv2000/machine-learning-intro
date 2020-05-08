#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:09:21 2020

@author: yuvrajsingh
"""


#Import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection (Build classic ANN)
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam' , loss= 'binary_crossentropy', metrics = ['accuracy'])

#Fitting images to CNN
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250, #Number of images being tested
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


from keras.preprocessing import image
import numpy as np
 
# Replace codo_2.jpg by your image path
 
test_image = image.load_img('dataset/test_image/download.jpg', target_size = (64, 64))
 
test_image = image.img_to_array(test_image)
 
test_image = np.expand_dims(test_image, axis = 0)
 
result = classifier.predict_classes(test_image)
 
print(result)
 
training_set.class_indices
 
if result[0][0] == 1:
    prediction = 'dog'    
    print("The image is of a ",prediction) 
else:
    prediction = 'cat'
    print("The image is of a ",prediction)

