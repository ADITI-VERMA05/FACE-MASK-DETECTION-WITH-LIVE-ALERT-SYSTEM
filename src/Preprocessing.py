import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(dataset_path="dataset"):
    
    categories = ["with_mask", "without_mask"]
    data = []
    labels = []

    for category in categories:
        path = os.path.join(dataset_path, category)
        label = categories.index(category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (100, 100))
            data.append(image)
            labels.append(label)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test