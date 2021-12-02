import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import os
import shutil
import cv2 as cv
from PIL import Image
import keras
from keras import models
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.applications.efficientnet import EfficientNetB0
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import initializers
from tensorflow.keras import optimizers
from keras.initializers import glorot_uniform
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.applications.resnet import ResNet50

from mtcnn.mtcnn import MTCNN

TRAIN = True
Re_save_data = False
use_mt_cnn = False
FAST_RUN = True

loss_function = 'binary_crossentropy'
# loss_function = 'categorical_crossentropy'
learning_rate = 0.01
momentum = 0.9
metrics = 'accuracy'


def affineMatrix(lmks, scale=2.5):
    nose = np.array(lmks['nose'], dtype=np.float32)
    left_eye = np.array(lmks['left_eye'], dtype=np.float32)
    right_eye = np.array(lmks['right_eye'], dtype=np.float32)
    eye_width = right_eye - left_eye
    angle = np.arctan2(eye_width[1], eye_width[0])
    center = nose
    alpha = np.cos(angle)
    beta = np.sin(angle)
    w = np.sqrt(np.sum(eye_width ** 2)) * scale
    m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
         [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]]
    return np.array(m), (int(w), int(w))  # （affine matrix, target size)


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def self_Resnet50(input_shape=(299, 299, 3), classes=2):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# Train a CNN classifier
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(299, 299, 3),
    include_top=False,
    pooling='max'
)


def create_model():
    model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()  # Determines the type of top model

    # Adds the convolutional model to the classifier
    for i in model.layers:
        top_model.add(i)
    # top_model.add(model)

    # formats the input for the classifier to the convolutional base output
    top_model.add(Flatten(input_shape=model.output_shape[1:]))

    # Condenses the input from flatten down to 256 nodes
    top_model.add(Dense(256, activation='relu'))

    top_model.add(Dropout(0.5))

    # Condenses the remaining input down into 2 categories
    top_model.add(Dense(1, activation='softmax', name='predictions'))

    top_model.compile(loss=loss_function,
                      optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=1e-6, nesterov=False),
                      # learning rate(lr) and momentum are Core Variables
                      # SGD options to consider: Nesterov, learning rate decay
                      metrics=[metrics])

    # top_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return top_model


def testing_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3  # R, G, B

deepfake_img = os.listdir("./fake_deepfake/")
face2face_img = os.listdir("./fake_face2face/")
real_img = os.listdir("./real/")
categories = []

if Re_save_data:
    if not os.path.exists("./training_data/"):
        os.makedirs("./training_data/")

    for filename in deepfake_img:
        print("Deepfake")
        original = "./fake_deepfake/" + filename
        target = './training_data/d_' + filename
        shutil.copyfile(original, target)
        categories.append(0)

    for filename in face2face_img:
        print("Face2Face")
        original = "./fake_face2face/" + filename
        target = './training_data/f_' + filename
        shutil.copyfile(original, target)
        categories.append(0)

    for filename in real_img:
        print("Real")
        original = "./real/" + filename
        target = './training_data/r_' + filename
        shutil.copyfile(original, target)
        categories.append(1)

training_img = os.listdir("./training_data/")

detector = MTCNN(steps_threshold=[0.0, 0.0, 0.0])

testing_folder = './testing/'

if use_mt_cnn:
    for i in training_img:
        print("processing ", i)
        img = cv.imread("./training_data/" + i)
        faces = detector.detect_faces(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        face = max(faces, key=lambda x: x['confidence'])
        mat, size = affineMatrix(face['keypoints'])

        tmp = cv.warpAffine(img, mat, size)
        cv.imwrite(testing_folder + i, tmp)

        img = Image.open(testing_folder + i)
        new_img = img.resize((256, 256))
        new_img.save(testing_folder + i)

# training_img = os.listdir("./testing/")
training_img = os.listdir("./training_data/")

categories = []
for i in training_img:
    tmp = i.split("_")[0]
    if tmp != "r":
        categories.append(str(0))
    else:
        categories.append(str(1))

df = pd.DataFrame({
    'filename': training_img,
    'category': categories
})

model = create_model()
model.summary()

# Prevent the model being overfit
earlystop = EarlyStopping(patience=10)

# Improve the learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            mode='auto')
                                            # factor=0.5,
                                            # min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=None, stratify=df["category"])
# train_df = train_df.reset_index(drop=True)
# validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 10

# =======================================================

# Training Set
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "./training_data/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,  # Default : 299, 299
    class_mode='binary',  # 2D numpy array of one-hot encoded labels. Supports multi-label output
    batch_size=batch_size
)

# =======================================================

# Validation Set
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "./training_data/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)

if TRAIN:
    epochs = 3 if FAST_RUN else 30

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate // batch_size,
        steps_per_epoch=total_train // batch_size,
        callbacks=callbacks
    )

    model.save_weights("./testing.h5")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))
    legend = ax1.legend(loc='best', shadow=True)

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    legend = ax2.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.savefig("./test.png")
    plt.show()

exit()
