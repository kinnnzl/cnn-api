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
                prediction = {"prediction": "Test"}
        return make_response(jsonify(prediction), 200)
