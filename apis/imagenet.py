from flask_restplus import Namespace, Resource
from PIL import Image
from werkzeug.datastructures import FileStorage

from core.utils import serve_pil_image

api = Namespace('imagenet', description='Models Trained on ImageNet')

upload_parser = api.parser()
upload_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)


@api.route('/resnet18/classify')
@api.expect(upload_parser)
class ResNet18Classify(Resource):
    def post(self):
        from core.models.resnet import ResNet18 # to save memory =(
        args = upload_parser.parse_args()
        uploaded_file = args['image']  # This is FileStorage instance
        image = Image.open(uploaded_file.stream)
        resnet18 = ResNet18()
        prediction = resnet18.predict(image)
        return prediction


@api.route('/superresolutionCL/transform')
@api.expect(upload_parser)
class SuperResolutionCLTransform(Resource):
    def post(self):
        from core.models.superresolutionCL import SuperResolutionCL
        args = upload_parser.parse_args()
        uploaded_file = args['image']
        image = Image.open(uploaded_file.stream)
        sr = SuperResolutionCL()
        hi_res = sr.transform(image)
        print(type(hi_res))
        return serve_pil_image(hi_res)

