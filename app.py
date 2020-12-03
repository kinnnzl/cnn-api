from flask import Flask
from flask_restful import Api

from resources.todo import GetMovies

app = Flask(__name__)
api = Api(app)

api.add_resource(GetMovies, "/GetMovies/")

if __name__ == "__main__":
  app.run()