from flask import Blueprint, jsonify
import os
from .helpers import *

cge_blueprint = Blueprint('cge_data', __name__)

@cge_blueprint.route('/cge_data/<int:ano>/<int:mes>/<int:dia>', methods=['GET'])
def get_all_ocurrencies(ano, mes, dia):
    directory = os.getcwd() + "/app/api/v1/cge_data/cge_json_files"
    
    json_file = load_specific_json(ano, mes, dia, directory)
    
    return jsonify(json_file)