from train import main
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
ml_seg = {
    'train': main
}

cors = CORS(app, resources={r'/{}'.format(module): {"origins": "*"} for module in ml_seg})

@app.route("/<string:service_name>", methods=["POST"])

def service(module):
    try: 
        ml_seg_module = ml_seg[module]
    except:
        return None, 401

    data = request.get_json()
    output_data = ml_seg_module(data)

    return jsonify(output_data)

if __name__ == "__main__":
    app.run(debug=True)