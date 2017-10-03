from flask_restplus import Namespace, Resource
import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage

from core.models import resnet


api = Namespace('imagenet', description='Models Trained on ImageNet')

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@api.route('/resnet18')
@api.expect(upload_parser)
class ResNet18(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file'] # This is FileStorage instance
        image = Image.open(uploaded_file.stream)
        resnet18 = resnet.ResNet18()
        prediction = resnet18.predict(image)
        return prediction

