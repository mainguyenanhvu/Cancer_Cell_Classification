from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, Input, ELU, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy
from keras import backend as K
import math
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# best_loss = 0.0

def data():
    original_dir = './training_seg_6'
    int_val_dir = './int_val_seg_6'
    img_width, img_height = 250, 250
    batch_size = 16

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       rotation_range=45,
                                       height_shift_range=0.2,
                                       width_shift_range=0.2,
                                       vertical_flip=True,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

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

    return train_generator, int_val_generator


def create_model(train_generator, int_val_generator):
    img_width, img_height = 250, 250
    channel_axis = 1 if K.image_dim_ordering() == "th" else -1
    epochs = 50
    batch_size = 16
    img_channels = 3
    classes_num = 9
    nb_train_samples = train_generator.samples
    nb_validation_samples = int_val_generator.samples
    log_dir = './test'

    model = Sequential()
    model.add(Conv2D({{choice([32, 64, 128])}}, kernel_size={{choice([2, 3, 4, 5])}},
                     input_shape=(img_width, img_height, img_channels), padding='same'))
    model.add(ELU(1.0))
    model.add(BatchNormalization(axis=channel_axis))

    model.add(Conv2D({{choice([32, 64, 128])}}, kernel_size={{choice([2, 3, 4, 5])}}, padding='same'))
    model.add(ELU(1.0))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size={{choice([2, 3])}}))

    model.add(Conv2D({{choice([64, 128, 256])}}, kernel_size={{choice([2, 3, 4, 5])}}, padding='same'))
    model.add(ELU(1.0))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size={{choice([2, 3])}}))

    model.add(Conv2D({{choice([64, 128, 256])}}, kernel_size={{choice([2, 3, 4, 5])}}, padding='same'))
    model.add(ELU(1.0))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size={{choice([2, 3, 4])}}))

    model.add(Conv2D({{choice([64, 128, 256])}}, kernel_size={{choice([2, 3, 4, 5])}}, padding='same'))
    model.add(ELU(1.0))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size={{choice([2, 3, 4])}}))

    model.add(Flatten())
    model.add(Dense({{choice([500, 750, 1000])}}))
    model.add(ELU(1.0))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # ý nghĩa sử dụng TensorBoard ???
    callback_log = TensorBoard(log_dir=log_dir,
                               histogram_freq=0,
                               batch_size=batch_size,
                               write_images=False,
                               write_grads=False,
                               write_graph=True)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs,
                                  validation_data=int_val_generator,
                                  validation_steps=nb_validation_samples / batch_size, verbose=2,
                                  callbacks=[callback_log],
                                  workers=32
                                  )

    loss = min(history.history['val_loss'])
    index_min = (history.history['val_loss']).index(loss)
    acc = (history.history['val_acc'])[index_min]
    # acc = min(history.history['val_acc'])

    # #only save model if it's the global best
    # path_best_model = './HO/XVII_test_seg6.keras'
    # global best_loss
    # if loss < best_loss:
    #     # save to hard disk
    #     model.save(path_best_model)
    #     # update the loss
    #     best_loss = loss

    # # clear the session and model
    # del model
    # K.clear_session()
    #
    print('Test accuracy:', acc)
    print('Test Loss', loss)
    print()
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    trials = Trials()
    train_generator, int_val_generator = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=75,
                                          trials=trials,
                                          )
    best_model.save('./HO/XVII_test_seg6.hdf5')
