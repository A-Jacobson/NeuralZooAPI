from flask import Flask
from flask_cors import CORS
from apis import api

app = Flask(__name__)
CORS(app)
api.init_app(app)

app.run(host='0.0.0.0', port=80)
# app.run(debug=True)
