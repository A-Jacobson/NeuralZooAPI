from PIL import Image
from flask_restplus import Namespace, Resource
from werkzeug.datastructures import FileStorage

from core.models.depthinthewild import DepthInTheWild
from core.utils import serve_pil_image

api = Namespace('nyudepth', description='Models Trained on NYUDepth')

upload_parser = api.parser()
upload_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)


@api.route('/depthinthewild/transform')
@api.expect(upload_parser)
class DepthInTheWildDepthTransform(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['image']
        image = Image.open(uploaded_file.stream)
        hourglass = DepthInTheWild()
        _, depth_map_img = hourglass.transform(image)
        return serve_pil_image(depth_map_img)


@api.route('/depthinthewild/transform_raw')
@api.expect(upload_parser)
class DepthInTheWildDepthTransformRaw(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['image']
        image = Image.open(uploaded_file.stream)
        hourglass = DepthInTheWild()
        depth_map, _ = hourglass.transform(image)
        return dict(depth_map=depth_map)
