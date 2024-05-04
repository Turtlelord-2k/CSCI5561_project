import os
import numpy as np
from PIL import Image
import keras
from keras.utils import Sequence

label_mapping = {
    0: 0,  # Unlabeled
    1: 0,  # Ego vehicle
    2: 0,  # Rectification border
    3: 0,  # Out of roi
    4: 0,  # Static
    5: 0,  # Dynamic
    6: 0,  # Ground
    7: 1,  # Road
    8: 1,  # Sidewalk
    9: 2,  # Parking
    10: 0,  # Rail track
    11: 3,  # Building
    12: 4,  # Wall
    13: 5,  # Fence
    14: 0,  # Guard rail
    15: 0,  # Bridge
    16: 0,  # Tunnel
    17: 6,  # Pole
    18: 0,  # Polegroup
    19: 7,  # Traffic light
    20: 8,  # Traffic sign
    21: 9,  # Vegetation
    22: 10,  # Terrain
    23: 11,  # Sky
    24: 12,  # Person
    25: 13,  # Rider
    26: 14,  # Car
    27: 15,  # Truck
    28: 16,  # Bus
    29: 0,  # Caravan
    30: 0,  # Trailer
    31: 17,  # Train
    32: 18,  # Motorcycle
    33: 19,  # Bicycle
}

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
            label = Image.open(label_path).resize((2048, 2048))

            image = np.array(image)
            # image = image/255.0
            image = np.transpose(image, (1, 0, 2))

            label = np.array(label)
            label = map_labels(label)  # Map labels to the reduced set of classes
            label = keras.utils.to_categorical(label, num_classes=20)  # Use 20 classes (19 + 1 for unknown)
            label = np.transpose(label, (1, 0, 2))

            batch_images.append(image)
            batch_labels.append(label)
        
        return np.array(batch_images), np.array(batch_labels)
    


def map_labels(labels):
    mapped_labels = np.zeros_like(labels)
    for k, v in label_mapping.items():
        mapped_labels[labels == k] = v
    
    # Set any remaining unexpected labels to a designated "unknown" class (19 in this case)
    mapped_labels[mapped_labels == labels] = 19
    
    return mapped_labels