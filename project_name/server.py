from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from backend.utils import transform_image, get_prediction

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/make-prediction', methods=['POST'])
@cross_origin(supports_credentials=True)
def make_prediction():
    imagefile = request.files.get('myimage', '')

    transformed_image = transform_image(imagefile.read())

    print('Transformed Image', transformed_image.shape)
    
    prediction = get_prediction(transformed_image)

    return jsonify({ "prediction": prediction }), 200
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')