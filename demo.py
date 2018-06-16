from keras.utils import np_utils
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(10)  # 使用tensorflow作为Backend
from keras.datasets import mnist

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()  # 加载图片与数据标签
# print(type(mnist.load_data()))
# print(y_test_label)
print(x_train_image.shape)
print(x_test_image.shape)
# print(x_train_image)
x_Train = x_train_image.reshape(60000, 28 * 28).astype('float32')
x_Test = x_test_image.reshape(10000, 28 * 28).astype('float32')
print(x_Train.shape)
print(x_Test.shape)
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255  # 数据标准化
print(y_train_label[0])
y_Train_One_Hot = np_utils.to_categorical(y_train_label)
print(y_Train_One_Hot[0])  # 看出第一个数据lable为5 所以第五个数字为1
y_Test_One_Hot = np_utils.to_categorical(y_test_label)
from keras.models import Sequential
from keras.layers import Dense  # 导入建立全连接层的包

model = Sequential()
# model.add(Dense(units=256,
#                 input_dim=784,
#                 kernel_initializer='normal',
#                 activation='relu')) # 在模型中添加

model.add(Dense(
    # input_shape=784,
    input_dim=784,
    units=256,
    kernel_initializer='normal',
    activation='relu'
))
# 在模型中添加 输入数据为784的神经元数量 也就是图拍呢一维的大小
# 隐藏层的神经元数量为256
# 　使用正态分布来随机初始化权重和偏差
# 激活函数使用的是relu
# model.add(Dense(
#     units=128,
#     activation='relu',
#     kernel_initializer='normal'
# ))
# model.add(Dense(
#     units=64,
#     activation='relu',
#     kernel_initializer='normal',
# ))
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_Train_normalize,
                          y=y_Train_One_Hot, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=2)
