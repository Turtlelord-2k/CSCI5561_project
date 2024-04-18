import os
import numpy as np
from PIL import Image
import keras
from keras.utils import Sequence

class CityscapesDataLoader(Sequence):
    def __init__(self, data_dir, batch_size, image_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = []
        self.label_paths = []
        
        # Get the paths of all the images and labels
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith("_leftImg8bit.png"):
                    image_path = os.path.join(root, file)
                    label_path = os.path.join(root.replace("leftImg8bit", "gtFine"), file.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
                    self.image_paths.append(image_path)
                    self.label_paths.append(label_path)
        
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_paths = self.label_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        batch_labels = []
        
        for image_path, label_path in zip(batch_image_paths, batch_label_paths):
            image = Image.open(image_path).resize(self.image_size)
            label = Image.open(label_path).resize(self.image_size)

            image = np.array(image)
            image = image/255.0

            label = np.array(label)
            label = keras.utils.to_categorical(label, num_classes=19)
            
            batch_images.append(image)
            batch_labels.append(label)
        
        return np.array(batch_images), np.array(batch_labels)
