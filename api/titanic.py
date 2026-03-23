from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.titanic import TitanicModel

titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class _Predict(Resource):
    def post(self):
        passenger = request.get_json()
        titanicModel = TitanicModel.get_instance()
        response = titanicModel.predict(passenger)
        return response

class _FeatureWeights(Resource):
    def get(self):
        titanicModel = TitanicModel.get_instance()
        response = titanicModel.feature_weights()
        return response

api.add_resource(_Predict, '/predict')                  # ← required
api.add_resource(_FeatureWeights, '/feature_weights')   # ← required