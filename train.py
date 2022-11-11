import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class Trainer():
    def __init__(self, X_data, Y_label, model_url, classes, epochs) -> None:
        self.epochs = epochs
        self.num_classes = classes
        self.X_data = X_data
        self.Y_label = Y_label
        self.IMAGE_SHAPE = (224, 224, 3)
        self.model_name = model_url
        self.model_top = hub.KerasLayer(
            model_url, input_shape=self.IMAGE_SHAPE, trainable=False
        )
    def train(self):
        self.model = tf.keras.Sequential([
            self.model_top, 
            tf.keras.layers.Dense(self.num_classes)
        ])
        self.model.summary()

        self.model.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['acc']
        )

        self.model.fit(self.X_data, self.Y_label, self.epochs)
    
    def predict(self, X_data):
        return self.model.predict(X_data)
    
    def evaluate(self, X_data, Y_label):
        return self.model.evaluate(X_data, Y_label)