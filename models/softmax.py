import tensorflow as tf
import numpy as np

# 获取数据
from tensorflow.keras.datasets import fashion_mnist

batch_size = 256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# 初始化模型参数
num_inputs = 28 * 28
num_outputs = 10
W = tf.Variable(tf.random.normal(shape=(num_inputs,num_outputs),stddev=0.01,dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=(num_outputs),dtype=tf.float32))

#定义模型
def softmax(logits, axis=-1):
    return tf.exp(logits)/tf.reduce_sum(tf.exp(logits), axis, keepdims=True)

def net(X):
    logits = tf.matmul(tf.reshape(X,shape=(-1,num_inputs)),W) + b
    return softmax(logits)

#定义损失函数
def cross_entropy(y_hat,y):
    '''
    计算交叉熵
    :param y_hat: np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    :param y:np.array([0, 2], dtype='int32')
    :return:
    '''
    y = tf.cast(tf.reshape(y,shape=(-1,1)),dtype=tf.int32)
    y = tf.one_hot(y,depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]),dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y)+1e-8)


#计算准确率
def accuracy(y_hat,y):
    np.mean(tf.argmax(y_hat,axis=1)==y)

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n


#训练模型
# 这里使用 1e-3 学习率，是因为原文 0.1 的学习率过大，会使 cross_entropy loss 计算返回 numpy.nan
num_epochs, lr = 5, 1e-3

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    global sample_grads
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = tf.reduce_sum(loss(y_hat, y))

            grads = tape.gradient(l, params)
            if trainer is None:

                sample_grads = grads
                params[0].assign_sub(grads[0] * lr)
                params[1].assign_sub(grads[1] * lr)
            else:
                trainer.apply_gradients(zip(grads, params))  # “softmax回归的简洁实现”一节将用到

            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat,    axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

trainer = tf.keras.optimizers.SGD(lr)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
