from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from backend.utils import transform_image, get_prediction

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#gets the image from the request.
#then transforms the image, makes a prediction and returns that prediction.
@app.route('/make-prediction', methods=['POST'])
@cross_origin(supports_credentials=True)
def make_prediction():

    try:
        imagefile = request.files.get('myimage', '')

        file_extension = imagefile.filename.rsplit('.', 1)[1].lower()
        if file_extension not in ['jpg', 'jpeg', 'png']:
            raise Exception('Wrong file type, only JPG, JPEG and PNG allowed', 400)
        
        transformed_image = transform_image(imagefile.read())

        print('Transformed Image', transformed_image.shape)
        
        prediction = get_prediction(transformed_image)

        return jsonify({ "prediction": prediction }), 200
    
    except Exception as e:
        if len(e.args) == 1:
            e.args = (e.args[0], 500)

        return jsonify({ "errorMessage": e.args[0] }), e.args[1]
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')