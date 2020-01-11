import tensorflow as tf

#加载数据集
from tensorflow.keras.datasets import fashion_mnist


(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

#定义初始化模型
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
model = Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(10,activation=tf.nn.softmax)
])

#定义损失函数

optimizer = tf.keras.optimizers.Adam(0.01)

#训练模型
model.compile(optimizer=optimizer,
              loss= tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=256)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Acc:',test_acc)

print(model.weights)


