#线性回归调用api直接实现
import tensorflow as tf


#生成数据集
from sklearn.datasets import load_iris
features,labels = load_iris(return_X_y=True)




#读取数据
batch_size = 10
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
dataset = dataset.batch(batch_size)



#定义模型，初始化参数
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers

model = keras.Sequential([
    layers.Dense(1)
])


#定义损失函数
from tensorflow import losses
loss = losses.MeanSquaredError()


#定义优化算法
optimizer = tf.keras.optimizers.Adam(0.01)

#训练模型
model.compile(optimizer=optimizer,
              loss= tf.losses.binary_crossentropy,
              metrics=['accuracy'])

model.fit(features,labels,epochs=5,batch_size=10)




print(model.get_weights())








