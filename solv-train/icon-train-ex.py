import tensorflow as tf
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from iconModel import IconModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# tf.compat.v1.enable_eager_execution()
# tf.enable_eager_execution()

train_txt = './icon-train-map.txt'
x_train_savepath = './datas/solv_icon_x_train.npy'
y_train_savepath = './datas/solv_icon_y_train.npy'
model_savepath = './models/solv-train/icon.ckpt'

def generateds(txt):
    f = open(txt, 'r')
    contents = f.readlines()
    # contents = f.readlines()[0:194]
    f.close()

    x, y_ = [], []
    for content in contents:
        value = content.split()
        img_path = value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))

        img = img / 255.

        x.append(img)
        y_.append(value[1])
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)

    return x, y_

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 24, 24))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_txt)
    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), 24, 24))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

x_data = x_train[:-798]
y_data = y_train[:-798]
x_test = x_train[-798:]
y_test = y_train[-798:]

x_data = x_data.reshape(x_data.shape[0], 24, 24, 1)
x_test = x_test.reshape(x_test.shape[0], 24, 24, 1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 255.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=5,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.2  # 将图像随机缩放阈量50％
)

image_gen_train.fit(x_data)

# 定义cnn网络
model = IconModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 存储模型
if os.path.exists(model_savepath + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(model_savepath)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_savepath,
    save_weights_only=True,
    save_best_only=True
)

# 未加形变的图像处理
history = model.fit(image_gen_train.flow(x_data, y_data, batch_size=32), epochs=200, validation_data=(x_test, y_test), callbacks=[cp_callback])
model.summary()

# 输出神经网络参数
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

image_path = "./icons/iconICON_Location&&3019.png"
img = Image.open(image_path)
img = img.resize((24, 24), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))

# for i in range(24):
#     for j in range(24):
#         if img_arr[i][j] < 200:
#             img_arr[i][j] = 255
#         else:
#             img_arr[i][j] = 0

img_arr = img_arr / 255.0
x_predict = img_arr.reshape(1, 24, 24, 1)
result = model.predict(x_predict)

pred = tf.argmax(result, axis=1)

print('\n')
# print(result)
tf.print(pred)

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
