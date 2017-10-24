from flask import Flask
from apis import api

app = Flask(__name__)
api.init_app(app)

app.run(host='0.0.0.0', port=80, debug=True)