import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from numpy import loadtxt
from keras.models import load_model
import os
import cv2
import numpy as np


from image_lib import *

train_files = []
directory = "images\\"
train_files = os.listdir(directory)

train_files = [x for x in train_files if x.endswith(".jpg")]

print(train_files)
train_files = shuffle(train_files)
print(train_files)
batch_size = 4
num_classes = 2
epochs = 12

img_rows, img_cols = 64, 64

images = np.zeros((len(train_files), 64, 64, 3), dtype="uint8")
classes = np.zeros((len(train_files)), dtype="uint8")

for index, file_name in enumerate(train_files):

    #print(directory + file_name)
    image = cv2.imread(directory + file_name)
    resized_image = cv2.resize(image, (64, 64))
    #cv2.imshow("", resized_image)
    #cv2.waitKey(5000)
    images[index] = resized_image
    if "jeans" in file_name:
        classes[index] = 1

train_size = 800
#(x_train, y_train), (x_test, y_test)
x_train = images[:train_size]
x_test = images[train_size:]
y_train = classes[:train_size]
y_test = classes[train_size:]

input_shape = (64, 64, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print('Test loss:', score[0])
print('Test accuracy:', score[1])