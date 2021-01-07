import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalMaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks
from keras.layers import LeakyReLU
from keras import backend as K
from keras.utils import data_utils
0

"""
import tensorflow as tf
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)

from keras import backend as K
K.set_session(sess)
"""
original_dir = './training_seg_6'
train_data_path = './training_seg_6'
validation_data_path = './int_val_seg_6'
ext_data_path = './ext_validation_seg_6'

"""
Parameters
"""
img_width, img_height = 250, 250
epochs = 100
batch_size = 64
classes_num = 9
img_channels = 3

model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(img_channels, img_width, img_height)))
model.add(Activation("relu"))
model.add(Conv2D(64, kernel_size=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(64, kernel_size=5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(128, kernel_size=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(128, kernel_size=3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(1000))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))
model.summary()

#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

ext_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

ext_validation_generator = ext_datagen.flow_from_directory(
    ext_data_path,
    target_size=(img_height, img_width),
    batch_size=1,
    shuffle = False,
    class_mode=None)

"""
Tensorboard log
"""

log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cbks = [tb_cb]

model.load_weights('models/Vreseg2_1st_weights.17-0.69.hdf5')

nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples

from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    ModelCheckpoint('./models/Vreseg5_1st_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2),
]

import tensorflow as tf
with tf.device('/gpu:2'):
        hist = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            verbose = 2,
            callbacks = callbacks)
print(hist.history)

y_predict = model.predict_generator(ext_validation_generator, verbose = 2)
model.save_weights('models/Vreseg2_2nd_weights.6-0.60.hdf5')
import numpy
from numpy import array
numpy.savetxt("y_predict-reseg2  b-3.csv", y_predict, delimiter=",")
filelist = ext_validation_generator.filenames
filelist = array(filelist)
numpy.savetxt("vallist-reseg2-3.csv", filelist, delimiter="/n", fmt='%s')

from keras.utils import plot_model
plot_model(model, to_file='model.png')
