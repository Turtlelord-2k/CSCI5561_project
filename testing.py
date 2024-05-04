import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from dataloader import CityscapesDataLoader
from elements import FasterSegModelFactory

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import UpSampling2D

from keras.models import Model

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

# Load the trained FasterSeg model
model_path = './model_21_04_23_24_02.h5'
model = FasterSeg(input_shape=(512, 1024, 3), num_classes=20)
model.load_weights(model_path)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the Cityscapes test dataset and randomly select an image
test_data_loader = CityscapesDataLoader(data_dir='./Cityscapes/leftImg8bit/test', batch_size=1, image_size=(512, 1024))
random_index = np.random.randint(len(test_data_loader))
image, _ = test_data_loader[random_index]

temp_image = image[0]
temp_image = cv2.transpose(temp_image)
temp_image = cv2.flip(temp_image, 1)

cv2.imshow('Image', temp_image)
cv2.waitKey(0)

# Resize the image to match the expected input shape
image = cv2.resize(image[0], (1024, 512))  # Resize to (width, height)

# Preprocess the selected image
image = image / 255.0

image = np.expand_dims(image, axis=0)  # Add the batch dimension

# Obtain the predicted segmentation mask
prediction = model.predict(image)
prediction = np.argmax(prediction, axis=-1)
prediction = prediction[0]  # Remove the batch dimension

# prediction = cv2.resize(prediction, (1024, 512), interpolation=cv2.INTER_NEAREST)


print("Unique predicted class labels:", np.unique(prediction))

class_colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]

class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign',
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle']

color_map = {i: tuple(color) for i, color in enumerate(class_colors)}

print("Color Map:")
for label, color in color_map.items():
    print(f"Label: {label}, Color: {color}")

# Create a blank RGB image for the color-labeled segmentation with the same shape as prediction
colored_segmentation = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

# Map the predicted class labels to their corresponding colors
for label, color in color_map.items():
    mask = prediction == label
    print(f"Label: {label}, Mask Shape: {mask.shape}, Num True: {np.sum(mask)}")
    colored_segmentation[mask] = color

# Resize the colored segmentation to match the shape of image[0]
# colored_segmentation = cv2.resize(colored_segmentation, (image.shape[2], image.shape[1]))

original_height, original_width = image[0].shape[:2]
colored_segmentation = cv2.resize(colored_segmentation, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

print("Image shape:", image.shape)
print("Colored segmentation shape:", colored_segmentation.shape)

# Convert the image and colored segmentation to float32
image = image.astype(np.float32) / 255.0
colored_segmentation = colored_segmentation.astype(np.float32) / 255.0

# Overlay the color-labeled segmentation on the original image
overlay = cv2.addWeighted(image[0], 0.7, colored_segmentation, 0.3, 0)

# Convert the overlay back to uint8 for display
overlay = (overlay * 255.0).astype(np.uint8)

overlay = cv2.transpose(overlay)
overlay = cv2.flip(overlay, 1)
print(overlay.shape)

# overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)



# Display the resulting image with the overlaid segmentation

cv2.imshow('Overlaid Segmentation', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# test_data_loader = CityscapesDataLoader(data_dir='./Cityscapes/leftImg8bit/test', batch_size=1, image_size=(512, 1024))
# test_loss, test_acc = model.evaluate(test_data_loader)
# print('Test accuracy:', test_acc)
# print('Test loss:', test_loss)