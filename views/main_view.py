from flask import Blueprint, request, jsonify, render_template
from ..utils import items_predictor

main = Blueprint('main', __name__)

items_predictor = items_predictor.ItemsPredictor()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    return items_predictor.predict(request.get_json())
