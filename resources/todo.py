from flask import jsonify, make_response
from flask_restful import Resource
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class CNN(Resource):
    def get(self, id):
        face_cascade = cv2.CascadeClassifier('C:/Users/User/Desktop/cnn/python-service-deployment/resources/haarcascade_frontalface_alt.xml')

        face_image = cv2.imread('C:/Users/User/Desktop/cnn/python-service-deployment/resources/3.2_5.jpg', 1)
        grey_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        plt.figure(figsize=(20, 10))
        plt.imshow(grey_img, cmap='gray')

        face_detect = face_cascade.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=5)
        print('Face found : ', len(face_detect))

        face_detect
        n = 25
        for (x, y, w, h) in face_detect:
            cv2.rectangle(face_image, (x, y), (x + w + n, y + h + n), (0, 255, 0), 3)
            crop_img = face_image[y:y + h + n, x:x + w + n]

        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('2_50.jpg', crop_img)

        # ------------------------------------------ cnn ------------------------------------------------------------

        # ========================================== IMAGE SIZE ==========================================
        width = 80
        height = 80
        n_classes = 5
        n_epochs = 50
        dropout_rate = 0.2

        # ========================================= TRAIN IMAGE ==========================================
        images_train = []
        labels_train = []

        for m in range(0, 1):
            path = "C:/Users/User/Desktop/cnn/python-service-deployment/resources/create model2/" + str(
                m) + "/train/*.jpg"
            for img in glob.glob(path):
                image = cv2.imread(img)
                image = cv2.resize(image, (width, height))
                image = img_to_array(image)
                images_train.append(image)
                label = m
                labels_train.append(label)
                # print (img)
                # print (label)

        # ==================================== TEST IMAGE =====================================
        images_test = []
        labels_test = []

        for m in range(0, 1):
            path = "C:/Users/User/Desktop/cnn/python-service-deployment/resources/create model2/" + str(
                m) + "/test/*.jpg"
            for img in glob.glob(path):
                image = cv2.imread(img)
                image = cv2.resize(image, (width, height))
                image = img_to_array(image)
                images_test.append(image)
                label = m
                labels_test.append(label)
                # print (img)
                # print (label)

        # ===================================== ARRAY =========================================
        # ======= PRINT TRAIN  =======
        train_images = np.array(images_train)
        train_labels = np.array(labels_train)
        # print (type(train_images))
        # print (type(train_labels))
        # print (train_images.shape)

        # ======= PRINT TEST =======
        test_images = np.array(images_test)
        test_labels = np.array(labels_test)
        # print (type(test_labels))
        # print (test_images.shape)

        # =================== Normalize pixel values to be between 0 and 1 ====================

        train_images, test_images = train_images / 255.0, test_images / 255.0

        model = models.Sequential()

        model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(width, height, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # model.summary()

        model.add(layers.Flatten())

        model.add(layers.Dense(32, activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(layers.Dense(64, activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(layers.Dense(n_classes, activation='softmax'))

        # model.summary()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=n_epochs);

        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print(test_acc)

        # ====================================== PREDICT ======================================
        predictions = model.predict(test_images)
        print(np.argmax(predictions, axis=1))
        print(test_labels)

        result = int(test_labels[0])
        prediction = {"prediction": result}
        return make_response(jsonify(prediction), 200)