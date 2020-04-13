#!/usr/bin/env python
# coding: utf-8

# # Neural Network for Detection of Fashion Item Images
# 
# ### Importing the libraries
# 
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# ## Preparing the Dataset
# 
# ### Importing data from keras : fashion_mnist
# 
# data=keras.datasets.fashion_mnist

# ### Splitting data into train and test
# 
# (train_images, train_labels),(test_images,test_labels)=data.load_data()

# ### Saving the class names of the items in the fasion_mnist data
# 
# class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal',
# 'Shirt','Sneaker','Bag','Ankle boot']

# ### Showing plots of the images
# 
# plt.imshow(train_images[4000])
# plt.show()

# ### Showing the real image
# 
# plt.imshow(train_images[4000], cmap=plt.cm.binary)

# ### RGB values of Images
# 
# print(train_images[4000])

# ### Standardizing the RGB Values in Images by dividing them by 255
# 
# train_images=train_images/255.0
# test_images=test_images/255
# 
# print(train_images[4000])

# ## Creating the Tensorflow Model
# 
# ### Step 1 Flattening the data which means creating a list of 784 values from 28X28 matrix of RGB values
# 
# ### By flatteining the data, we are creating a list of 784 neurons as input layer to be fed in Neural Net
# 
# ### Model1 that carries 128 neurons' hidden layer and it will be fully connected with input and output layer
# 
# model1=keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128,activation="relu"),
#     keras.layers.Dense(10,activation='softmax')
#     ])
# 
# model1.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 
# model1.fit(train_images, train_labels, epochs=10)
# 
# test_loss, test_acc=model1.evaluate(test_images,test_labels)
# 
# print("Tested Acc:", test_acc)

# ### Prediction with the model
# 
# prediction=model1.predict(test_images)
# 
# print(prediction)
# 
# 

# In[23]:


print(prediction[0])


# In[27]:


print(class_names[np.argmax(prediction[5])])

