import tensorflow as tf
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from iconModel import IconModel

# tf.compat.v1.enable_eager_execution()
# tf.enable_eager_execution()

icons_dir_path = "d:/code/python/tensorflow-learn/solv-train/icons"
train_txt = './icon-train-map.txt'
x_train_savepath = './datas/solv_icon_x_train.npy'
y_train_savepath = './datas/solv_icon_y_train.npy'
model_savepath = './models/solv-train/icon.ckpt'
model_savepath_export = './models/solv-train/saved'

def generateds(txt):
    f = open(txt, 'r')
    contents = f.readlines()
    
    # contents = f.readlines()[0:194]
    f.close()

    x, y_ = [], []
    for content in contents:
        value = content.split()
        img_path = icons_dir_path + "/" + value[0]
        img = Image.open(img_path)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img = np.array(img.convert('L'))

        img = img / 255.

        x.append(img)
        y_.append(value[1])
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)

    np.random.seed(116)
    np.random.shuffle(x)
    np.random.seed(116)
    np.random.shuffle(y_)
    tf.compat.v1.random.set_random_seed(116)

    return x, y_

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_txt)
    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), 28, 28))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

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
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.3, validation_freq=2, callbacks=[cp_callback])
model.summary()

model.save(model_savepath_export, save_format="tf")

# 输出神经网络参数
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

image_path = "./icons/iconICON_Alert1&&5372.png"
img = Image.open(image_path)
img = img.resize((28, 28), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))

# for i in range(24):
#     for j in range(24):
#         if img_arr[i][j] < 200:
#             img_arr[i][j] = 255
#         else:
#             img_arr[i][j] = 0

img_arr = img_arr / 255.0

# x_predict = img_arr[tf.newaxis, ..., 1]
x_predict = img_arr.reshape(1, 28, 28, 1)
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


# 转换模型
# tensorflowjs_converter \
#     --input_format=tf_saved_model \
#     --output_format=tfjs_graph_model \
#     --signature_name=serving_default \
#     --saved_model_tags=serve \
#     D:/code/python/tensorflow-learn/solv-train/models/solv-train/saved \
#     D:/code/python/tensorflow-learn/solv-train/models/solv-train/convert