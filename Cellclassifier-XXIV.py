import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, Input, ELU, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
import tensorflow as tf
import numpy
from keras import backend as K
import math
from keras import initializers

#IX : Batch, ReLu, Dropout = 0.25
#XI: Batch, ELu, Dropout = 0.25
#XII: BED(0.1)
#XIIV: BED(0.5)
#XIV: BED(0.5),modify adam
#XV: ED(0.5),modify adam
#XVI: ED(0.5),modify adam. correcting Batch?
#XVII: BED(0.5), Batch to Flatten
#XVIII: double dense stack
#XIV: double dense, remove Batch
#XX: remove batch, dense = 500 => adjust batchsize to 16
#XXIII: HO
#current: XXIV
channel_axis = 1 if K.image_dim_ordering() == "th" else -1
channel_axis = 1
cpuset = 0 #1 or other
if cpuset == 1:
    cpulist = ["gpu(2)", "gpu(3)"]
else:
    cpulist = ["gpu(0)", "gpu(1)"]


img_width, img_height = 250, 250
epochs = 200
batch_size = 16
img_channels = 3
classes_num = 9

train_data_path = './training_seg_5'
validation_data_path = './int_validation_seg_5'
ext_data_path = './ext_validation_seg_6'

original_dir = './training_seg_6'
int_val_dir = './int_val_seg_6'
int_test_dir = './int_test_seg_6'

#custom filter
def filter_layer(x):
    red_x = x[:,:,:,0]
    blue_x = x[:,:,:,2]
    green_x = x[:,:,:,1]
    red_x = tf.expand_dims(red_x, axis=3)
    blue_x = tf.expand_dims(blue_x, axis=3)
    green_x = tf.expand_dims(green_x, axis=3)
    #output = tf.concat([red_x, blue_x], axis=3)
    output = green_x
    return output

#model
input = Input(shape=(img_channels, img_height, img_width))
#x = Lambda(filter_layer)(input)
x = Conv2D(64, (5, 5), padding='same')(input)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)

x = Conv2D(64, (2, 2), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)

x = Conv2D(256, (2, 2), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (5, 5), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)

x = Conv2D(128, (4, 4), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(4, 4))(x)

x = Flatten()(x)
x = Dense(500)(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
#x = Dense(500)(x)
#x = ELU(alpha=1.0)(x)
#x = BatchNormalization(axis=channel_axis)(x)

x = Dropout(0.5)(x)

output = Dense(classes_num, activation='softmax')(x)
model = Model(inputs=input, outputs=output)
model.summary()

#model = multi_gpu_model(model, 4)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              context=cpulist)

# generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=45,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   vertical_flip=True,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# #without val
# train_generator = train_datagen.flow_from_directory(
#     train_data_path,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_data_path,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical')
#
# ext_validation_generator = test_datagen.flow_from_directory(
#     ext_data_path,
#     target_size=(img_height, img_width),
#     batch_size=1,
#     shuffle=False,
#     class_mode=None)
#
# nb_train_samples = train_generator.samples
# nb_validation_samples = validation_generator.samples


# withval
train_generator = train_datagen.flow_from_directory(
    original_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

int_val_generator = test_datagen.flow_from_directory(
    int_val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

int_test_generator = test_datagen.flow_from_directory(
    int_test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

nb_train_samples = train_generator.samples
nb_validation_samples = int_val_generator.samples

ext_validation_generator = test_datagen.flow_from_directory(
    ext_data_path,
    target_size=(img_height, img_width),
    batch_size=1,
    shuffle=False,
    class_mode=None)


"""
#withval
train_generator = train_datagen.flow_from_directory(
    original_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

int_val_generator = test_datagen.flow_from_directory(
    int_val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

int_test_generator = test_datagen.flow_from_directory(
    int_test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

nb_train_samples = train_generator.samples
nb_validation_samples = int_val_generator.samples
"""

#load weight
#model.load_weights('./weight/XVII-4-cont-200-reseg6_weights.85-0.37.hdf5')

#callback
from keras import callbacks
from keras.callbacks import TensorBoard
import numpy as np

class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


#callbacks = [TensorBoardWrapper(validation_generator, nb_steps=nb_validation_samples // batch_size, log_dir='./tf-log', histogram_freq=1,
                              # batch_size=int(batch_size), write_graph=False, write_grads=True)]

from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    #EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    ModelCheckpoint('./weight/XXIV-3-200-reseg6_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2)
    ,CSVLogger('./log/XXIV-3-200-reseg6-log.csv',append=False, separator=',')
    # ,TensorBoardWrapper(int_val_generator, nb_steps=math.ceil(nb_validation_samples / batch_size), log_dir='./tf-log-XXIV-rg6/',
    #                    histogram_freq=5,
    #                    batch_size=int(batch_size), write_graph=True, write_grads=False, write_images=False)
            ]

#train
history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              epochs=epochs,
                              validation_data=int_val_generator,
                              validation_steps=math.ceil(nb_validation_samples / batch_size),verbose=2,
                              callbacks=callbacks,
                              workers=32
                              )
hist = history.history
model.save_weights('./weight/XXIII-3-50-reseg6_weights.hdf5')
model.evaluate_generator(int_test_generator)
loss = np.array(hist["loss"])
y_predict = model.predict_generator(ext_validation_generator, verbose = 2)
#export result
numpy.savetxt("./result/XVII-predict-4-0.37-reseg6.csv", y_predict, delimiter=",")
numpy.savetxt("./log/XXIII-1-200-reseg6-log-loss-2.csv", loss, delimiter=",")


"""
filelist = ext_validation_generator.filenames
filelist = array(filelist)
numpy.savetxt("./result/vallist-IX-reseg5.csv", filelist, delimiter="/n", fmt='%s')
"""
