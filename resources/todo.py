import base64
from flask import jsonify, make_response, json, request
from flask_restful import Resource
import matplotlib.pyplot as plt
import numpy as np
import cv2


class CNN(Resource):
    def post(self):
        r = json.dumps(request.get_json(silent=True))
        encoded_image = r.split(",")[1]
        img = base64.b64decode(encoded_image)
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)

        # face_cascade = cv2.CascadeClassifier('https://github.com/kinnnzl/cnn-api/blob/master/resources'
        #                                      '/haarcascade_frontalface_alt.xml')
        #
        # grey_img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        #
        # plt.figure(figsize=(20, 10))
        # plt.imshow(grey_img, cmap='gray')
        #
        # face_detect = face_cascade.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=5)
        # print('Face found : ', len(face_detect))
        #
        # face_detect
        # n = 25
        # for (x, y, w, h) in face_detect:
        #     cv2.rectangle(source, (x, y), (x + w + n, y + h + n), (0, 255, 0), 3)
        #     crop_img = source[y:y + h + n, x:x + w + n]
        #
        # plt.figure(figsize=(20, 10))
        # plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        # cv2.imwrite('2_501.jpg', crop_img)

        # ------------------------------------------ cnn ------------------------------------------------------------

        # ========================================== IMAGE SIZE ==========================================

        # return make_response(jsonify(prediction), 200)
        return make_response(jsonify("prediction"), 200)
