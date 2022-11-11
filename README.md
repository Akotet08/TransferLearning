# TransferLearning

### Objective
To implement a machine-learning model that can classify miner and rust diseases in coffee plants. The easiest way to build such a model is to use transfer learning. As such, I will be using two pre-trained models from Tensorflow-hub, namely MobileNetV2 and InceptionV3. We will compare the performance of these two models.

### The dataset
I used a dataset available here on Kaggle. It consists of 257 images of coffee plants with miners(Leucoptera coffeella) and 285 images of plants with rust(Hemileia vastatrix). The original resolution is 4000x2250 pixels. The plants are of the species Coffea arabica.

### Training and Test split
80% for the training set and 20% for the test set. 

