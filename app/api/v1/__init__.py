from flask import Blueprint

api_v1_blueprint = Blueprint('api_v1', __name__)

from .cge_data import cge_blueprint
from .modelo_previsao import modelo_blueprint

api_v1_blueprint.register_blueprint(cge_blueprint, url_prefix='/cge_data')
api_v1_blueprint.register_blueprint(modelo_blueprint, url_prefix='/modelo_previsao')

