import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

tf.enable_eager_execution()

model_savepath = './models/solv-train/icon.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(236, activation="softmax")
])

model.load_weights(model_savepath)

image_path = "./test.png"

img = Image.open(image_path)
img = img.resize((24, 24), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))

img_arr = img_arr / 255.
x_predict = img_arr[tf.newaxis, ...]
result = model.predict(x_predict)

pred = tf.argmax(result, axis=1)

print('\n')
print(pred)
