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
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D, Resizing, LeakyReLU, \
    Concatenate, MaxPool2D
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
from keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy

from mtcnn.mtcnn import MTCNN


def my_custom_loss(y_true, y_pred):
    crossentropy = binary_crossentropy(y_true, y_pred)
    return crossentropy


def create_model():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_EfficientNet():
    model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                           pooling='max')
    print('Model loaded.')

    top_model = Sequential()
    top_model.add(model)
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(units=512, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(units=128, activation='relu'))
    top_model.add(Dense(units=1, activation='sigmoid'))
    top_model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return top_model


def create_VGG16():
    # model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # # model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # print('Model loaded.')
    #
    # # build a classifier model to put on top of the convolutional model
    # top_model = Sequential()  # Determines the type of top model
    #
    # top_model.add(model)
    #
    # # formats the input for the classifier to the convolutional base output
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #
    # # Condenses the input from flatten down to 256 nodes
    # top_model.add(Dense(256, activation='relu'))
    #
    # top_model.add(Dropout(0.5))
    #
    # # Condenses the remaining input down into 2 categories
    # top_model.add(Dense(1, activation='softmax', name='predictions'))
    #
    # top_model.compile(loss=loss_function,
    #                   optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=1e-6, nesterov=False),
    #                   # learning rate(lr) and momentum are Core Variables
    #                   # SGD options to consider: Nesterov, learning rate decay
    #                   metrics=[metrics])
    #
    # # top_model.compile(loss=loss_function, optimizer='Adam', metrics=['accuracy'])
    # return top_model

    model = Sequential()
    model.add(Conv2D(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), filters=64, kernel_size=(3, 3), padding="same",
                     activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(GlobalAveragePooling2D())
    model.add(GlobalMaxPooling2D())
    # model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(1, activation="softmax"))

    model.compile(loss=loss_function, optimizer='Adam', metrics=['accuracy'])
    return model


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])  # SKIP Connection
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def create_ResNet50():
    # model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # print('Model loaded.')
    #
    # # build a classifier model to put on top of the convolutional model
    # top_model = Sequential()  # Determines the type of top model
    #
    # top_model.add(model)
    #
    # # formats the input for the classifier to the convolutional base output
    # # top_model.add(GlobalAveragePooling2D())
    #
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #
    # # Condenses the input from flatten down to 256 nodes
    # top_model.add(Dense(256, activation='relu'))
    #
    # top_model.add(Dropout(0.5))
    #
    # top_model.add(Dense(128, activation='relu'))
    #
    # # Condenses the remaining input down into 2 categories
    # top_model.add(Dense(1, activation='softmax', name='predictions'))
    #
    # top_model.compile(loss=loss_function,
    #                   optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=1e-6, nesterov=False),
    #                   # learning rate(lr) and momentum are Core Variables
    #                   # SGD options to consider: Nesterov, learning rate decay
    #                   metrics=[metrics])
    #
    # # top_model.compile(loss=loss_function, optimizer='Adam', metrics=['accuracy'])
    # return top_model

    X_input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    base_model = Model(inputs=X_input, outputs=X, name='ResNet50')

    headModel = base_model.output
    headModel = Flatten()(headModel)
    headModel = Dropout(0.25)(headModel)
    headModel = Dense(256, activation='relu', name='fc1', kernel_initializer=glorot_uniform(seed=0))(headModel)
    headModel = Dropout(0.25)(headModel)
    headModel = Dense(128, activation='relu', name='fc2', kernel_initializer=glorot_uniform(seed=0))(headModel)
    headModel = Dense(1, activation='sigmoid', name='fc3', kernel_initializer=glorot_uniform(seed=0))(headModel)

    model = Model(inputs=base_model.input, outputs=headModel)

    # model.compile(loss="mean_squared_error",
    #               optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=1e-6, nesterov=False),
    #               # learning rate(lr) and momentum are Core Variables
    #               # SGD options to consider: Nesterov, learning rate decay
    #               metrics=[metrics])

    model.compile(loss=my_custom_loss, optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

    return model


def test_model():
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
    y = Dropout(0.25)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.25)(y)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=x, outputs=y)
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

    return model


def create_mesoInception4():
    def InceptionLayer(a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])

            return y

        return func

    x = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    x1 = InceptionLayer(1, 4, 4, 2)(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = InceptionLayer(2, 4, 4, 2)(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    y = Flatten()(x4)
    y = Dropout(0.25)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.25)(y)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=x, outputs=y)

    model.compile(loss=my_custom_loss, optimizer=optimizers.Adam(0.001), metrics=['accuracy'])

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


def preprocessing(deepfake_dir, face2face_dir, real_dir, training_dir, Re_save=False, mt_cnn=False):
    deepfake_img = os.listdir(deepfake_dir)
    face2face_img = os.listdir(face2face_dir)
    real_img = os.listdir(real_dir)

    if Re_save:
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)

        for filename in deepfake_img:
            print("Deepfake")
            original = deepfake_dir + filename
            target = training_dir + 'deepfake_' + filename
            shutil.copyfile(original, target)

        for filename in face2face_img:
            print("Face2Face")
            original = face2face_dir + filename
            target = training_dir + 'face2face_' + filename
            shutil.copyfile(original, target)

        for filename in real_img:
            print("Real")
            original = real_dir + filename
            target = training_dir + 'real_' + filename
            shutil.copyfile(original, target)

    if mt_cnn:
        mtcnn_dir = './testing/'
        mtcnn(mtcnn_dir, training_dir)
        training_img = os.listdir("./testing/")
    else:
        training_img = os.listdir("./training_data/")

    categories = []
    for i in training_img:
        tmp = i.split("_")[0]
        if tmp == "deepfake":
            categories.append("fake")
        elif tmp == "face2face":
            categories.append("fake")
        else:
            categories.append("real")

    df = pd.DataFrame({
        'filename': training_img,
        'category': categories
    })

    return df


def generate_data_set(df, batch_size):
    # df["category"] = df["category"].replace({0: 'fake', 1: 'real'})
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=None, stratify=df["category"])

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

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
        class_mode='binary',
        batch_size=batch_size
    )

    print(train_generator.class_indices)

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

    print(validation_generator.class_indices)

    return train_generator, validation_generator, total_train, total_validate


def model_training(model, train_data, validate_data, batch_size, num_train, num_validation):
    # Prevent the model being overfit
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')

    # Improve the learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=5 ,
                                                verbose=1,
                                                mode='auto',
                                                factor=0.75,
                                                min_lr=0.000001)

    callbacks = [earlystop, learning_rate_reduction]

    total_train = num_train
    total_validate = num_validation
    batch_size = batch_size

    if TRAIN:
        epochs = 3 if FAST_RUN else 150

        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=validate_data,
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


TRAIN = True
Re_save_data = False
use_mt_cnn = False
FAST_RUN = False
batch_size = 25
loss_function = 'binary_crossentropy'
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
training_dir = "./training_data/"

# model = create_model()
# model = create_VGG16()

# =====================================
model = create_ResNet50()
# create_ResNet50_V1
# stopped at epoch 100, loss: 0.0212 - accuracy: 0.9912 - val_loss: 0.1275 - val_accuracy: 0.9571 - lr: 1.5625e-05
# (Use earlystop = loss and ReduceLROnPlateau = accuracy for callbacks)

# =====================================
# model = create_EfficientNet()
# =====================================

# model = create_mesonet()
# stopped at epoch 21, loss: 0.1781 - accuracy: 0.7299 - val_loss: 0.1650 - val_accuracy: 0.7492

# model = create_mesoInception4()
# create_mesoInception4_V1
# stopped at epoch 6, loss: 0.5624 - accuracy: 0.7181 - val_loss: 0.5480 - val_accuracy: 0.7100

# create_mesoInception4_V2
# stopped at epoch 62, loss: 0.1816 - accuracy: 0.9269 - val_loss: 0.1709 - val_accuracy: 0.9275 - lr: 1.0000e-05
# (Use earlystop = loss and ReduceLROnPlateau = accuracy for callbacks)

# create_mesoInception4_V3
# stopped at epoch 45, loss: 0.2723 - accuracy: 0.8804 - val_loss: 0.2110 - val_accuracy: 0.9125 - lr: 1.2500e-04
# (Use earlystop = val_loss and ReduceLROnPlateau = val_accuracy for callbacks)

# =====================================

# model = test_model()

model.summary()

df = preprocessing(deepfake_dir, face2face_dir, real_dir, training_dir, Re_save_data, use_mt_cnn)

train_set, validation_set, num_train, num_validation = generate_data_set(df, batch_size)

model_training(model, train_set, validation_set, batch_size, num_train, num_validation)
exit()
