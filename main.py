import os

from flask import Flask, json, request, send_from_directory
import random

from Network import predict

vector = [{"id": 1, "state": "Healthy"}, {"id": 2, "state": "Unhealthy"}]

api = Flask(__name__)

@api.route('/health', methods=['GET'])
def get_companies():
    chosen = random.choice(vector)
    print(chosen)
    return json.dumps(chosen)

api.config['UPLOAD_PATH'] = 'files'

if not os.path.exists(api.config['UPLOAD_PATH']):
    os.makedirs(api.config['UPLOAD_PATH'])

@api.route('/files', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    print(uploaded_file.filename)
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(api.config['UPLOAD_PATH'], uploaded_file.filename + ".nii"))
    predict()
    return "", 201

# @api.route("/files")
# def list_files():
#     """Endpoint to list files on the server."""
#     files = []
#     for filename in os.listdir(UPLOAD_DIRECTORY):
#         path = os.path.join(UPLOAD_DIRECTORY, filename)
#         if os.path.isfile(path):
#             files.append(filename)
#     return jsonify(files)

@api.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(api.config['UPLOAD_PATH'], path, as_attachment=True)

if __name__ == "__main__":
    # api.run(debug=False, port=8000,host= '127.0.0.1')
    api.run(debug=False, port=8000,host= '192.168.0.80')