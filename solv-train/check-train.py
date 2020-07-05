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

image_path = "./test-icons/iconBrowser_Chrome&&2319.png"

img = Image.open(image_path)
img = img.resize((24, 24), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))

img_arr = img_arr / 255.
# x_predict = img_arr[tf.newaxis, ...]
x_predict = img_arr.reshape(1, 24, 24, 1)
result = model.predict(x_predict)

pred = tf.argmax(result, axis=1)

print('\n')
print(pred)
