import numpy as np
import time
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import UpSampling2D

from keras.models import Model

from elements import *
from dataloader import *


EPOCHS = 30
BATCH_SIZE = 12
VERBOSE = 1
ts = time.strftime('%d_%m_%H_%M_%S')
checkpoint_filepath = f"model_{ts}.h5"

train_data_loader = CityscapesDataLoader(data_dir='./Cityscapes/leftImg8bit/train', batch_size=BATCH_SIZE, image_size=(512, 1024))
val_data_loader = CityscapesDataLoader(data_dir='./Cityscapes/leftImg8bit/val', batch_size=BATCH_SIZE, image_size=(512, 1024))

def FasterSeg(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    stem = FasterSegModelFactory.stem_module(inputs)

    branch_out_16 = []
    branch_out_32 = []

    operators_list = [['3x3_conv_x2', '3x3_conv_x2'], ['zoomed_conv_3x3_x2'], ['zoomed_conv_3x3'],
                    ['zoomed_conv_3x3_x2', 'zoomed_conv_3x3_x2'], ['zoomed_conv_3x3_x2', 'zoomed_conv_3x3_x2'],
                    ['zoomed_conv_3x3_x2'], ['zoomed_conv_3x3_x2', 'zoomed_conv_3x3_x2']]
    
    expansion_ratios = [4, 4, 8, 4, 4, 4, 4, 4, 8]

    for cell_id, (operators, er) in enumerate(zip(operators_list, expansion_ratios)):
        downsample_rate = 2**(cell_id + 1)
        x_16, x_32 = FasterSegModelFactory.cell_module([stem, stem], downsample_rate, er, operators)
        branch_out_16.append(x_16)
        branch_out_32.append(x_32)

    output_16 = FasterSegModelFactory.head_module(branch_out_16)
    output_32 = FasterSegModelFactory.head_module(branch_out_32)

    output = Concatenate()([output_16, output_32])
    output = Conv2D(num_classes, kernel_size=3, padding='same')(output)
    output = UpSampling2D(size=(4,2))(output)
    output = tf.keras.layers.Softmax()(output)

    model = Model(inputs, output)
    return model

# tf.config.run_functions_eagerly(True)

model = FasterSeg(input_shape=(512, 1024, 3), num_classes = 20)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()])

checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss')


history = model.fit(train_data_loader, validation_data=val_data_loader, epochs=EPOCHS, callbacks=[checkpoint_callback])



epochs = history.epoch
Training_loss = history.history['loss']
Training_accuracy = history.history['accuracy']
Val_loss = history.history['val_loss']
Val_accuracy = history.history['val_accuracy']

plt.plot(Training_accuracy)
plt.plot(Val_accuracy)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


plt.plot(Training_loss)
plt.plot(Val_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()