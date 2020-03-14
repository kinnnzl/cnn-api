from flask import jsonify, make_response
from flask_restful import Resource



class CNN(Resource):
    def get(self, id):
        prediction = {"prediction": "Test"}
        return make_response(jsonify(prediction), 200)
