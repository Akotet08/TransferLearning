import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import pandas as pd
from sklearn.model_selection import train_test_split
from train import Trainer
import seaborn as sn
import tensorflow as tf


def process_data(): 

    IMAGE_SHAPE = (224, 224)

    suf = '.jpg'

    miner_dir = './datasets/miner_images/' # 257 images
    rust_dir = './datasets/rust_images/' # 285 images

    miner = [miner_dir + f'bicho_mineiro{j}' + suf for j in range(258)]
    rust = [rust_dir + f'name{k+1}' + suf for k in range(324) ]

    coffee_images_dict = {
        'miner': miner,
        'rust': rust
    }

    coffee_labels_dict = {
        'miner': 0, 
        'rust':1
    }

    X_data, Y_labels = [], []

    for cls, images in coffee_images_dict.items():
        for image in images:
            try:
                img = cv2.imread(image)
                img = cv2.resize(img, IMAGE_SHAPE)
                X_data.append(img)
                Y_labels.append(coffee_labels_dict[cls])
            except Exception as e:
                # print(image)
                pass

    X_data = np.array(X_data)
    Y_labels = np.array(Y_labels)

    return X_data, Y_labels

def plot_confusion_matrix(prediction, true_label):
    in_mat = tf.math.confusion_matrix(prediction, true_label)
    in_mat = np.array(in_mat).astype('int32')

    in_df = pd.DataFrame(in_mat, index = ['miner', 'rust'], columns=['miner', 'rust'])

    plt.figure(figsize = (10,7))
    plt.title('Confusion matrix for the InceptionV3-based Model')
    s = sn.heatmap(in_df, annot=True)
    s.set(xlabel='Predicted Labels', ylabel='Real Labels')

if __name__== '__main__':

    X_data, Y_labels = process_data()

    mobilenet = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    inception = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

    X_data = np.array(X_data)
    Y_labels = np.array(Y_labels)

    x_train, x_test, y_train, y_test  = train_test_split(X_data, Y_labels, test_size=0.2, 
                                                        train_size=0.8, shuffle=True, stratify=Y_labels)

    mobilenet = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    inception = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

    incept_trainer = Trainer(x_train, y_train, inception, classes= 2, epochs=7)
    mobile_trainer = Trainer(x_train, y_train, mobilenet, classes= 2, epochs=7)

    incept_trainer.train()
    mobile_trainer.train()

    incept_prediction = incept_trainer.predict(x_test)
    incept_predictions = np.array([np.argmax(prd) for prd in incept_prediction])

    mobile_prediction = mobile_trainer.predict(x_test)
    mobile_predictions = np.array([np.argmax(pred) for pred in mobile_prediction])


