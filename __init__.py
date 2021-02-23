from flask import Flask
from flask_cors import CORS

from .views.main_view import main

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

app.register_blueprint(main, url_prefix = '/')

if __name__ == "__main__":

    print("Server up!")

    app.run()
