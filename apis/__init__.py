from flask_restplus import Api

from .imagenet import api as imagenet
from .nyudepth import api as nyudepth
from .mscoco import api as mscoco
from .imdb import api as imdb

api = Api(
    title='NeuralZoo API',
    version='1.0',
    description='API for Deep Learning models',

    # All API metadatas
)

api.add_namespace(imagenet)
api.add_namespace(mscoco)
api.add_namespace(nyudepth)
api.add_namespace(imdb)

