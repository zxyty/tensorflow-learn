import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

tf.enable_eager_execution()

# 导入数据，分别为输入特征核标签

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据
seed = 115
np.random.seed(seed)
np.random.shuffle(x_data)
np.random.seed(seed)
np.random.shuffle(y_data)
# tf.random.set_random_seed(seed)
tf.compat.v1.random.set_random_seed(seed)

# print(tf.__version__)

x_train = x_data[:-30]
y_train = y_data[:-30]

x_test = x_data[-30:]
y_test = y_data[-30:]

# print(x_train, y_train)
# print(x_test, y_test)

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 32组数据定义为一个batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

# 学习率
lr = 0.1
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0

# 训练
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        # 这里会遍历4次
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)    # 激活函数

            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))    # 均方误差 loss
            loss_all += loss.numpy()
        
        # 计算loss 对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])
        print(step)
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    # print("After %s epoch, loss is: %f" %(epoch, loss_all / 4))
    train_loss_results.append(loss_all / 4)
    loss_all = 0

    # 测试部分
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 其实这里只遍历1次
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)    # 激活函数
        
        pred = tf.argmax(y, axis=1) # 返回y中最大值的索引 即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)

        # 若分类正确则返回bool类型，将bool类型转换为int类型0 , 1
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)

        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试总样本数
        total_number += int(x_test.shape[0])

    acc = total_correct / total_number
    test_acc.append(acc)

    # print("test acc: %f" %(acc))

plt.title("Loss Function")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()


plt.title("Acc Result")
plt.xlabel("Epoch")
plt.ylabel("Acc")

plt.plot(test_acc, label="$Acc$")
plt.legend()
plt.show()