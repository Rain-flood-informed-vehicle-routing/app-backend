from flask import Flask
from flask import Blueprint
from flask_cors import CORS

from .api.v1 import api_v1_blueprint

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api_v1_blueprint, url_prefix='/api/v1')
    return app