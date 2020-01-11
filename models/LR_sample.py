#线性回归调用api直接实现
import tensorflow as tf


#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal(shape=(num_examples, num_inputs), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(labels.shape, stddev=0.01)


#读取数据
batch_size = 10
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
dataset = dataset.shuffle(buffer_size=num_examples)
dataset = dataset.batch(batch_size)


#定义模型，初始化参数
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers

model = keras.Sequential()
model.add(layers.Dense(1,kernel_initializer=initializers.RandomNormal(stddev=0.01)))

#定义损失函数
from tensorflow import losses
loss = losses.MeanSquaredError()

#定义优化算法
from tensorflow.keras import optimizers
trainer = optimizers.SGD(learning_rate=0.01)

#训练模型
num_epochs = 3
for epoch in range(1,num_epochs+1):
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape(persistent=True) as t:
            l = loss(model(X,training=True),y)

        grads = t.gradient(l,model.trainable_variables)
        trainer.apply_gradients(zip(grads,model.trainable_variables))

    l = loss(model(features),labels)
    print('epoch %d, loss: %f' % (epoch, l.numpy().mean()))



print(model.get_weights())






