from PIL import Image
from flask_restplus import Namespace, Resource
from werkzeug.datastructures import FileStorage

from core.utils import serve_pil_image

api = Namespace('mscoco', description='Models Trained on MSCOCO')

upload_parser = api.parser()
upload_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)


@api.route('/styletransfer_starry/transform')
@api.expect(upload_parser)
class StyleTransferStarryTransform(Resource):
    def post(self):
        from core.models.faststyletransfer import StyleTransfer
        args = upload_parser.parse_args()
        uploaded_file = args['image']  # This is FileStorage instance
        image = Image.open(uploaded_file.stream)
        st = StyleTransfer('starry')
        styled_img = st.transform(image)
        return serve_pil_image(styled_img)


@api.route('/styletransfer_udnie/transform')
@api.expect(upload_parser)
class StyleTransferUndieTransform(Resource):
    def post(self):
        from core.models.faststyletransfer import StyleTransfer
        args = upload_parser.parse_args()
        uploaded_file = args['image']
        image = Image.open(uploaded_file.stream)
        st = StyleTransfer('udnie')
        styled_img = st.transform(image)
        return serve_pil_image(styled_img)
