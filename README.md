# Deep Fake Detection
Experiment, Study and Result
Model
The are some model that have been studied for this project, namely "MesoNet", "MesoInception4", "VGG16", "EfficientNet" and "InceptionResNetV2". However, as I am using my own GPU for training (RTX2070 Super), it only contains 8GB gpu-ram and therefore may not have enough time to handle "InceptionResNetV2", as they have very deep structure.
For the tested model, the parameter including the learning rate, earlystopping and learning reduced are fine turned many times for each model, only the best result of each model will be given.

============================== Tested ==============================
MesoNet
The the MesoNet is a shallow convolutional neural network which comprises a sequence of four convolution layers and pooling and is followed by a fully connected dense layer with one hidden layer in between. The convolutional layers use ReLu as activation function and baatch normalization to regularize the output, in which, it can preven the vanishing gradient problem. The dropout applied at the fully-connected layers can improve the robustness and take generalization on another level.
image-18.png

image.png from https://arxiv.org/pdf/2106.12605.pdf (Deep Fake Detection : Survey of Facial Manipulation Detection Solutions)

==============================

In this project, the result provided by MesNet is not as good as other and here is the best result:

"loss: 0.1781 - accuracy: 0.7299 - val_loss: 0.1650 - val_accuracy: 0.7492".

(Use "earlystop = val_loss" and "ReduceLROnPlateau = val_accuracy" for callbacks, lr = 0.001)

==============================

MesoInception4
The MesoInception4 is a model that based on MesoNet, it used a InceptionLayer to replace all single convolutional layer in MesNet.
image-15.png

image-19.png

==============================

In this project, the result provided by MesoInception4 is quite good and here is the best result:

"loss: 0.1816 - accuracy: 0.9269 - val_loss: 0.1709 - val_accuracy: 0.9275"

(Use" earlystop = loss" and "ReduceLROnPlateau = accuracy" for callbacks, lr = 0.001)

==============================

VGG16
The VGG16 is a simple and widely used Convolutional Neural Network, it only contain 5 convolutional block
image-24.png from https://neurohive.io/en/popular-networks/vgg16/

With refer to the configurations list, I had implemented config-C for experiment, however, the units of dense(fully-connect) layer has reduced to 1024 and 2048 due to the memory of my GPU. If I keep using 4096 units, my GPU will not have enough memory and cause an error. A self-defined top level layer have added to the model.

image-50.png

image-25.png from https://arxiv.org/abs/1409.1556 (Very Deep Convolutional Networks for Large-Scale Image Recognition)

The result of the VGG16 is quite weird which the accuracy does not improve after any epoch, many fine turning have be done to VGG16, and even pre-trained weight has also been tested, however, the result still the same. I had already sent many email to TAs, but none of their suggestion works :

" loss: 0.6368 - accuracy: 0.6667 - val_loss: 0.6368 - val_accuracy: 0.6667 "

(Use" earlystop = loss" and "ReduceLROnPlateau = val_accuracy" for callbacks, lr = 0.0001)

image-51.png

EfficientNet B0, B3, B5, B7
EfficientNet B0 is a model implentented using AutoML MNA, one of the objective of EfficientNet is to increase the efficient of training, compare with other tested model, its performance is amazing ! The training accuracy comes to ~90% with less than 10 epochs, which the other tested network need more than 15~20 epoch under the same learning rate. For the EfficientNet B1-B7, they are just scaled up version of B0, which included "width scaling", "Depth Scaling", "Resolution Scaling" and "Compound scaling".
image-45.png from https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

The EfficientNet model is created using the library of keras, and self created the top layer for it :

image-21.png

The structure of EfficientNet B0: image-20.png from https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

==============================

In this project, B0, B3, B5 and B7 share the same structure of top level layer : image-35.png

the best result provided by EfficientNet B0 :

"loss: 0.0192 - accuracy: 0.9934 - val_loss: 0.2740 - val_accuracy: 0.9317"

(Use" earlystop = loss" and "ReduceLROnPlateau = val_accuracy" for callbacks, lr = 0.0001)

the best result provided by EfficientNet B3 :

"loss: 0.0141 - accuracy: 0.9950 - val_loss: 0.7142 - val_accuracy: 0.7404"

(Use" earlystop = loss" and "ReduceLROnPlateau = val_accuracy" for callbacks, lr = 0.0001)

the best result provided by EfficientNet B5 :

"loss: 0.0055 - accuracy: 0.9978 - val_loss: 0.1523 - val_accuracy: 0.9638"

(Use" earlystop = loss" and "ReduceLROnPlateau = val_accuracy" for callbacks, lr = 0.0001)

the best result provided by EfficientNet B7 :

"loss: 0.0050 - accuracy: 0.9984 - val_loss: 0.0719 - val_accuracy: 0.9804"

(Use" earlystop = loss" and "ReduceLROnPlateau = val_accuracy" for callbacks, lr = 0.0001)

==============================

ResNet50
The "ResNet" is a kind of CNN structure introduced by Kaimei He in 2016 (Deep Residual Learning for Image Recognition ), it prevoide the ability to create a very deep network with a low probability of happening "Gradient Vanishing" or "Degradation problem". In the past, when the layer of a network become very deep, it will affect the model's learning power and cannot be improved in certain iteration. And this is the result why "ResNet" is choosen. In "ResNet", the important thing is identity transform, which is a shortcut simply speaking. The shortcut will cross over the current layer components and become the second next layer's input and activation function. The "ResNet50" model use Adam as the optimizer which is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments, and the loss is set to be binary crossentropy due to the face that this project is a binary classfication problem (fake or real faces).
image-30.png

identity_block : The identity block is the standard block used in ResNets, and corresponds to the case where the input activation has the same dimension as the output activation. The upper path is the "shortcut path." The lower path is the "main path." In this diagram, we have also made explicit the CONV2D and ReLU steps in each layer. To speed up training we have also added a BatchNorm step. When the shortcut is used, it will skip connection "skips over" 3 hidden layers.

image-33.png

image-27.png

convolutional_block : The convolutional block is another type of block which is used when there are different bewteen input and output dimensions, there contain a Conv2D layer in the shortcut which is different from the identity block. The Conv2D layer in shortcut aims to resize the input to other dimension in order to much up with the final addition needed and to add the shortcut value back to the main path. Moreover, the reason why it does not use any activation function is due to the fact that it only aims to reduces the dimension of input.

image-34.png

image-28.png

The structure of ResNet50 is built following the structure introduced in the paper "Deep Residual Learning for Image Recognition". The identity shortcut can directly be used in the circumstances of input and output have the same dimensions (solid line). And there are two options why the dimensions increase (dotted line), the first one is that a extrz zero padding is applied for increasing the dimensions, which the shortcut still performs identity mapping; The second one is using a linear projection by the shorcut connections to match the dimensions (Kaiming, Xiangyu, Shaoqing, Jian, 2016)

image-32.png

image-31.png

image-29.png

==============================

In this project, the result provided by ResNet50 is very good and here is the best result:

"loss: 0.0155 - accuracy: 0.9964 - val_loss: 0.1017 - val_accuracy: 0.9746" (Use" earlystop = loss" and "ReduceLROnPlateau = val_accuracy" for callbacks, lr = 0.0002)

==============================

============================== Untested ==============================
InceptionResNetV2
The InceptionResNetV2 is a model that combine the concept of "shortcuts" from ResNet and Inception module, it aims to increase the depth or network to give a more accurate prediciton.
it use 1x1 convolution without activation to scaling up the dimensionality of the filter bank before the addition to match the depth of the input. This is needed to compensate for the dimensionality reduction induced by the Inception block.

The graph below is a simple structure of InceptionResNetV2:

image-8.png

Stem
The schema for stem of the pure Inception-ResNet-v2 networks:
image-5.png

=============== Inception-ResNet ===============
The Inception-ResNet module used a asymmetry convolutional layer, it also used 1x1 convolution without activation to scaling up the dimensionality of the filter bank before the addition to match the depth of the input. This is needed to compensate for the dimensionality reduction induced by the Inception block.
Inception-ResNet-A
The schema for 35 × 35 grid module of the Inception-ResNet-v2 network:
image-9.png

Inception-ResNet-B
The schema for 17 × 17 grid module of the Inception-ResNet-v2 network:
image-11.png

Inception-ResNet-C
The schema for 8×8 grid module of the Inception-ResNet-v2 network:
image-13.png

=============== Reduction ===============
The reduction module is used to slowly decrease the size of feature map slowly, in order to reduce the loss of feature information.
Reduction-A
The schema for 35 × 35 to 17 × 17 reduction module of Inception-ResNet-v2 network:
image-10.png

Reduction-B
The schema for 17 × 17 to 8 × 8 grid-reduction module of the Inception-ResNet-v2 network:
image-12.png

Activation ScalingC
The general schema for scaling combined Inceptionresnet moduels: ![image-14.png](attachment:image-14.png)
from https://arxiv.org/pdf/1602.07261.pdf (Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning)

Model Chosen
From the above result, the ResNet50 and EfficientNet-B7 has the highest accuracy which is 0.9964 and 0.9984. Therefore the final model will be chosen between them. Before selection, there are few things need to be consider other than the training accuracy, which is the generalization error, it is kind of presenting the how accurate the model perform with the unseen data. To review the generalization capabilities, val_loss and val_accuracy need to be measure :

ResNet50 val_loss: 0.1017 - val_accuracy: 0.9746

EfficientNetB7 val_loss: 0.0719 - val_accuracy: 0.9804

The result show that EfficientNet-B7 perform better than the ResNet50, however, the val_loss and val_accuracy of EfficientNet is more unstable than the ResNet50, from the graph below, it is easy to see that the val_loss and val_accuracy of EfficientNet is more fluctuating than the ResNet50

ResNet50 :

image-38.png image-39.png

EfficientNet-B7 :

image-40.png image-41.png

The reason why the EfficientNet-B7 has a more fluctuating val_loss and val_accuracy may due to its large scale of network and overfitting. The EfficientNet_B7 is finally be chosen because it have the highest training and validation accuracy. But the fluction problem cannot be ignore due to the fact that it may affect the reason of testing. Some modification one early stopping and learning rate have been tested, but it does not give a satisfy reason. Therefore, some modification has been done to the top level layer of the EfficientNet_B7:

The orignial top level layer flatten the output of model and added a flatten layer followed with a fully connect layer, then a drop out is added followed with two fully connect layer. One of the method suggested by Mr Hanwei is to use drop out to improve model generalization capabilities.

Before : image-42.png

After : image-43.png

However, the result does not improve much which the val_loss and val_accuracy still very fluctuate

image-46.png

Therefore, another method suggested by Mr Hanwei have been added on top of the dropout appoach, which is regularization term - L2 norm for the fully connected layer.

image-49.png

After introduced the L2 norm, the result did improved in later training preiod and especially for the validation loss. Moreover, there are tiny improvement in the performance. And the final model for this project is based on such approach.

image-48.png

image-47.png
