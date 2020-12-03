import pandas as pd
import json
from flask import Flask, jsonify
from flask_restful import Resource


class GetMovies(Resource):
    def post(self):
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
        ratings = pd.merge(movies, ratings).drop(['genres', 'timestamp'], axis=1)
        print(ratings.shape)

        user_ratings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
        user_ratings.head()
        print("Before: ", user_ratings.shape)
        user_ratings = user_ratings.dropna(thresh=10, axis=1).fillna(0, axis=1)
        print("After: ", user_ratings.shape)

        corr_matrix = user_ratings.corr(method='pearson')
        # print(corr_matrix.sum().sort_values(ascending=False))

        # def get_similar(movie_name, rating):
        #     similar_ratings = corrMatrix[movie_name].rating * (rating - 2.5)
        #     similar_ratings = similar_ratings.sort_values(ascending=False)
        #     print(type(similar_ratings))
        #     return similar_ratings

        # Return the response in json format
        result = corr_matrix.sum().sort_values(ascending=False).to_json(orient="table")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        return make_response(jsonify(parsed), 200)
