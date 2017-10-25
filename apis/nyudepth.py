from flask_restplus import Namespace, Resource
from PIL import Image
from werkzeug.datastructures import FileStorage

from core.models import depthinthewild

api = Namespace('nyudepth', description='Models Trained on NYUDepth')

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@api.route('/depthinthewild')
@api.expect(upload_parser)
class DepthInTheWild(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        image = Image.open(uploaded_file.stream)
        hourglass = depthinthewild.DepthInTheWild()
        depth_map, depth_map_img = hourglass.predict(image)
        return depth_map, depth_map_img
