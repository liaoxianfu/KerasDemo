from keras.utils import np_utils
from keras.datasets import mnist

# import numpy
(train_img, train_lable), (test_img, test_labe) = mnist.load_data()
# print(train_img.shape)

# print(train_lable.shape)
train_img_reshape = train_img.reshape(train_img.shape[0], 28 * 28).astype('float32')
test_img_reshape = test_img.reshape(test_img.shape[0], 28 * 28).astype('float32')
# print(train_img_reshape.shape)
train_img_reshape_normalize = train_img_reshape / 255.0
test_img_reshape_normalize = test_img_reshape / 255.0
# print(test_img_reshape_normalize)
train_lable_one_hot = np_utils.to_categorical(train_lable)
# print(train_lable_one_hot)
test_labe_ont_hot = np_utils.to_categorical(test_labe)

# 建立模型与层
from keras.models import Sequential  # 导入全连接层
from keras.layers import Dense, Dropout  # 导入层模型

model = Sequential()
model.add(
    Dense(input_dim=28 * 28,
          units=1000,
          kernel_initializer='normal',
          activation='relu'
          )
)

model.add(Dropout(0.5))
model.add(
    Dense(
        units=256,
        kernel_initializer='normal',
        activation='relu'
    )
)
model.add(Dropout(0.5))
model.add(
    Dense(units=50,
          kernel_initializer='normal',
          activation='relu'
          )
)
model.add(Dropout(0.5))
model.add(
    Dense(
        units=10,
        kernel_initializer='normal',
        activation='softmax'
    )
)
from keras import losses
from keras import optimizers

model.compile(
    loss=losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']

)

train_history = model.fit(x=train_img_reshape_normalize,
                          y=train_lable_one_hot, validation_split=0.2,
                          epochs=30, batch_size=500, verbose=2)


scores = model.evaluate(test_img_reshape_normalize,test_labe_ont_hot)
print()
print(scores[1])
