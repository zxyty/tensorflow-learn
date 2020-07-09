import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from iconModel import IconModel

# tf.enable_eager_execution()

model_savepath = './models/solv-train/icon.ckpt'

model = IconModel()

model.load_weights(model_savepath)

image_path = "./test-icons/Snipaste_2020-07-06_16-09-46.png"

img = Image.open(image_path)
img = img.resize((28, 28), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))

img_arr = img_arr / 255.
# x_predict = img_arr[tf.newaxis, ...]
x_predict = img_arr.reshape(1, 28, 28, 1)
result = model.predict(x_predict)

pred = tf.argmax(result, axis=1)

print('\n')

print(pred)

# np.argsort(np.ndarray.flatten(result))[-1:-10]
top10PredIndex = np.argsort(np.ndarray.flatten(result))[-10::, ].tolist()
top10PredIndex.reverse()

print(top10PredIndex)

