from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import ReLU
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
# from keras.layers import DownSampling2D
from keras.layers import AveragePooling2D

class FasterSegModelFactory():
    
    @staticmethod
    def stem_module(img_size):
        #Stem Module
        # img_h, img_w = img_size.shape()
        # input_size = Input(shape=(img_h, img_w, 3))
        stem = Conv2D(32, 3, strides=2, padding='same')(img_size)
        stem = BatchNormalization()(stem)
        stem = ReLU()(stem)

        stem = Conv2D(64, 3, strides=2, padding='same')(img_size)
        stem = BatchNormalization()(stem)
        stem = ReLU()(stem)

        stem = Conv2D(64, 3, strides=1, padding='same')(img_size)
        stem = BatchNormalization()(stem)
        stem = ReLU()(stem)

        stem = Conv2D(128, 3, strides=2, padding='same')(img_size)
        stem = BatchNormalization()(stem)
        stem = ReLU()(stem)

        stem = Conv2D(128, 3, strides=1, padding='same')(img_size)
        stem = BatchNormalization()(stem)
        stem = ReLU()(stem)

        return stem
    
    def cell_module(inputs, downsample_rate, expansion_ratio, operators):
        input_1, input_2 = inputs

        for op in operators:
            if op == 'skip_connect':
                cell = input_1
            elif op == '3x3_conv':
                cell = Conv2D(int(downsample_rate * expansion_ratio), 3, padding='same')(input_1)
                cell = BatchNormalization()(cell)
                cell = ReLU()(cell)
            elif op == '3x3_conv_x2':
                cell = Conv2D(int((downsample_rate * expansion_ratio)//2), 3, padding='same')(input_1)
                cell = BatchNormalization()(cell)
                cell = ReLU()(cell)
                cell = Conv2D(int((downsample_rate * expansion_ratio)), 3, padding='same')(cell)
                cell = BatchNormalization()(cell)
                cell = ReLU()(cell)
            elif op == 'zoomed_conv_3x3':
                cell = UpSampling2D()(input_1)
                cell = BatchNormalization()(cell)
                cell = ReLU()(cell)
                cell = Conv2D(int((downsample_rate * expansion_ratio)), 3, padding='same')(cell)
                cell = BatchNormalization()(cell)
                cell = ReLU()(cell)
                cell = AveragePooling2D()(cell)
            elif op == 'zoomed_conv_3x3_x2':
                cell = UpSampling2D()(input_1)
                cell = Conv2D(int((downsample_rate * expansion_ratio)//2), 3, padding='same')(input_1)
                cell = BatchNormalization()(cell)
                cell = ReLU()(cell)
                cell = Conv2D(int((downsample_rate * expansion_ratio)), 3, padding='same')(cell)
                cell = BatchNormalization()(cell)
                cell = ReLU()(cell)
                cell = AveragePooling2D()(cell)

            input_1 = cell

        if input_1.shape[1] > input_2.shape[1] or input_1.shape[2] > input_2.shape[2]:
            # Downsample input_1 to match the spatial dimensions of input_2
            pool_size = (input_1.shape[1] // input_2.shape[1], input_1.shape[2] // input_2.shape[2])
            input_1 = AveragePooling2D(pool_size=pool_size)(input_1)
        elif input_1.shape[1] < input_2.shape[1] or input_1.shape[2] < input_2.shape[2]:
            # Upsample input_1 to match the spatial dimensions of input_2
            up_size = (input_2.shape[1] // input_1.shape[1], input_2.shape[2] // input_1.shape[2])
            input_1 = UpSampling2D(size=up_size)(input_1)
        
        input_2 = Conv2D(input_1.shape[-1], 1, padding='same')(input_2)
        input_2 = BatchNormalization()(input_2)
        input_2 = ReLU()(input_2)

        cell = Add()([input_1, input_2])

        if downsample_rate > 8:
            output_1 = Conv2D(int((downsample_rate * expansion_ratio)), 3, padding='same')(cell)
            output_1 = BatchNormalization()(output_1)
            output_1 = ReLU()(output_1)
        else:
            output_1 = cell

        output_2 = cell

        return [output_1, output_2]
    

    def head_module(inputs):

        input_1 = inputs[0]   # lower res feature map
        input_2 = inputs[1]   # higher res feature map      

        head = Conv2D(filters = input_1.shape[-1], kernel_size=1)(input_2)
        head = BatchNormalization()(head)
        head = ReLU()(head)

        if input_1.shape[1] < head.shape[1] or input_1.shape[2] < head.shape[2]:
            up_size = (head.shape[1] // input_1.shape[1], head.shape[2] // input_1.shape[2])
            input_1 = UpSampling2D(size=up_size)(input_1)

        head = Concatenate()([input_1, head])

        head = Conv2D(filters=input_1.shape[-1], kernel_size=3, padding='same')(head)
        head = BatchNormalization()(head)
        head = ReLU()(head)

        return head



