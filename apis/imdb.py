from flask_restplus import Namespace, Resource


api = Namespace('imdb', description='Models Trained on Imdb/Rotten Tomatoes')

upload_parser = api.parser()
upload_parser.add_argument('text', required=True)


@api.route('/textcnn/classify')
@api.expect(upload_parser)
class TextCNNClassify(Resource):
    def get(self):
        from core.models.textcnn import TextCNNWrapper # to save memory =(
        args = upload_parser.parse_args()
        text = args['text']  # This is FileStorage instance
        textcnn = TextCNNWrapper()
        probability, prediction = textcnn.classify(text)
        return dict(prediction=prediction, probability=probability)
