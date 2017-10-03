from flask_restplus import Api

from .imagenet import api as imagenet

api = Api(
    title='ZooAPI',
    version='1.0',
    description='Machine Learning API for DL models',
    # All API metadatas
)

api.add_namespace(imagenet)