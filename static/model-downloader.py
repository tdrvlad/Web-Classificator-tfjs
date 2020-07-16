import sys
import os
import tensorflow as tf
import tensorflowjs as tfjs

path = './tfjs-models/'

#Download VGG19 
model_name = 'VGG19'

if os.path.exists(path + model_name):
    print(model_name + ' model already exists')
else:
    print('Downloading and saving ' + model_name + ' model')
    model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    tfjs.converters.save_keras_model(model, path + model_name)

#Download MobileNet 
model_name = 'MobileNet'

if os.path.exists(path + model_name):
    print(model_name + ' model already exists')
else:
    print('Downloading and saving ' + model_name + ' model')
    model = tf.keras.applications.MobileNet()
    tfjs.converters.save_keras_model(model, path + model_name)

