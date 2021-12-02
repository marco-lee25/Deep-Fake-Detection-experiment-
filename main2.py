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
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D, Resizing, LeakyReLU
from keras.applications.efficientnet import EfficientNetB0
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import initializers, layers
from tensorflow.keras import optimizers
from keras.initializers import glorot_uniform
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.utils.layer_utils import get_source_inputs
from keras_vggface.vggface import VGGFace
from sklearn.model_selection import train_test_split
from keras.applications.resnet import ResNet50
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.utils import to_categorical

from mtcnn.mtcnn import MTCNN


def create_model():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_EfficientNet():
    model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()  # Determines the type of top model

    top_model.add(model)

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

    # top_model.compile(loss=loss_function, optimizer='Adam', metrics=['accuracy'])
    return top_model


def create_VGG16():
    model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()  # Determines the type of top model

    top_model.add(model)

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

    # top_model.compile(loss=loss_function, optimizer='Adam', metrics=['accuracy'])
    return top_model


def create_ResNet50():
    model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()  # Determines the type of top model

    top_model.add(model)

    # formats the input for the classifier to the convolutional base output
    top_model.add(Flatten(input_shape=model.output_shape[1:]))

    # Condenses the input from flatten down to 256 nodes
    top_model.add(Dense(256, activation='relu'))

    top_model.add(Dropout(0.5))

    top_model.add(Dense(128, activation='relu'))

    # Condenses the remaining input down into 2 categories
    top_model.add(Dense(1, activation='softmax', name='predictions'))

    top_model.compile(loss=loss_function,
                      optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=1e-6, nesterov=False),
                      # learning rate(lr) and momentum are Core Variables
                      # SGD options to consider: Nesterov, learning rate decay
                      metrics=[metrics])

    # top_model.compile(loss=loss_function, optimizer='Adam', metrics=['accuracy'])
    return top_model


def test_model():
    # # model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # model = VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # # model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # print('Model loaded.')
    #
    # # build a classifier model to put on top of the convolutional model
    # top_model = Sequential()  # Determines the type of top model
    #
    # top_model.add(model)
    #
    # for layer in top_model.layers[:17]:
    #     layer.trainable = False
    #
    # # 22 layer for VGG19
    #
    # for layer in top_model.layers[17:]:
    #     layer.trainable = True
    #
    # last_layer = model.get_layer('block5_pool')
    # last_output = last_layer.output
    #
    # x = GlobalMaxPooling2D()(last_output)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    #
    # x = layers.Dense(2, activation='softmax')(x)
    # model = Model(model.input, x)
    #
    # # model.compile(loss=loss_function,
    # #                   optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=1e-6, nesterov=False),
    # #                   # learning rate(lr) and momentum are Core Variables
    # #                   # SGD options to consider: Nesterov, learning rate decay
    # #                   metrics=[metrics])
    #
    # model.compile(loss=loss_function, optimizer='Adam', metrics=['accuracy'])
    # return model
    vgg_model = VGGFace(include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    last_layer = vgg_model.get_layer('pool5').output

    # model = GlobalMaxPooling2D()(last_layer)
    model = Dense(256, activation='relu', name="Marco_1")(last_layer)
    model = Dropout(0.5)(model)
    model = Flatten(name='flatten')(model)
    model = Dense(512, activation='relu', name='fc1')(model)
    model = Dense(1, activation='sigmoid', name='dense2')(model)

    custom_vgg_model = Model(vgg_model.input, model)
    custom_vgg_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002), metrics=['acc'])

    return custom_vgg_model


def create_mesonet():
    x = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    x1 = Resizing(224, 224, interpolation='bilinear', crop_to_aspect_ratio=False)(x)
    x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=x, outputs=y)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002), metrics=['accuracy'])

    return model


def mtcnn(mtcnn_dir, training_dir):

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

    detector = MTCNN(steps_threshold=[0.0, 0.0, 0.0])
    for i in training_dir:
        print("processing ", i)
        img = cv.imread(training_dir + i)
        faces = detector.detect_faces(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        face = max(faces, key=lambda x: x['confidence'])
        mat, size = affineMatrix(face['keypoints'])

        tmp = cv.warpAffine(img, mat, size)
        cv.imwrite(mtcnn_dir + i, tmp)

        img = Image.open(mtcnn_dir + i)
        new_img = img.resize((256, 256))
        new_img.save(mtcnn_dir + i)


def preprocessing(deepfake_dir, face2face_dir, real_dir, training_dir, Re_save=True, mt_cnn=False):
    deepfake_img = os.listdir(deepfake_dir)
    face2face_img = os.listdir(face2face_dir)
    real_img = os.listdir(real_dir)

    count_d = 0
    count_f = 0
    count_r = 0

    if Re_save:
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            os.makedirs(training_dir + "train/fake")
            os.makedirs(training_dir + "train/real")
            os.makedirs(training_dir + "valid/fake")
            os.makedirs(training_dir + "valid/real")

        for filename in deepfake_img:
            print("Deepfake")
            original = deepfake_dir + filename
            if count_d % 5 != 0 :
                target = training_dir + "train/fake/" + 'deepfake_' + filename
                shutil.copyfile(original, target)
            else:
                target = training_dir + "valid/fake/" + 'deepfake_' + filename
                shutil.copyfile(original, target)
            count_d += 1

        for filename in face2face_img:
            print("Face2Face")
            original = face2face_dir + filename
            if count_f % 5 != 0 :
                target = training_dir + "train/fake/" + 'face2face_' + filename
                shutil.copyfile(original, target)
            else:
                target = training_dir + "valid/fake/" + 'deepfake_' + filename
                shutil.copyfile(original, target)
            count_f += 1

        for filename in real_img:
            print("Real")
            original = real_dir + filename
            if count_r % 5 != 0 :
                target = training_dir + "train/real/" + 'real_' + filename
                shutil.copyfile(original, target)
            else:
                target = training_dir + "valid/real/" + 'deepfake_' + filename
                shutil.copyfile(original, target)
            count_r += 1




def generate_data_set(batch_size, training_dir):
    dataset_path = training_dir

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
    train = train_datagen.flow_from_directory(dataset_path + 'train/',
                                              class_mode="binary",
                                              target_size=IMAGE_SIZE,
                                              batch_size=batch_size)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    valid = validation_datagen.flow_from_directory(
        dataset_path + 'valid/',
        target_size=IMAGE_SIZE,
        class_mode='binary',
        batch_size=batch_size
    )


    return train, valid


def model_training(model, train_data, valid_data, batch_size):
    # Prevent the model being overfit
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

    # Improve the learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                mode='auto')
                                                # factor=0.5,
                                                # min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]

    total_train = len(train)
    total_validate = len(valid_data)
    batch_size = batch_size

    if TRAIN:
        epochs = 3 if FAST_RUN else 30

        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=valid_data,
            validation_steps=total_validate ,
            steps_per_epoch=total_train ,
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



TRAIN = True
Re_save_data = False
use_mt_cnn = False
FAST_RUN = False
batch_size = 25
loss_function = 'categorical_crossentropy'
learning_rate = 0.01
momentum = 0.9
metrics = 'accuracy'

IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3  # R, G, B

deepfake_dir = "./fake_deepfake/"
face2face_dir = "./fake_face2face/"
real_dir = "./real/"
training_dir = "./training_data2/"


# model = create_VGG16()
# model = create_ResNet50()
# model = create_EfficientNet()
# model = create_mesonet
model = test_model()

model.summary()

preprocessing(deepfake_dir, face2face_dir, real_dir, training_dir, Re_save_data, use_mt_cnn)


train, valid = generate_data_set(batch_size, training_dir)


model_training(model, train, valid, batch_size)









