import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='public')

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Define the upload folder
# UPLOAD_FOLDER = '/uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/public/<path:path>')
def serve_static_files(path):
    return send_from_directory('public', path)


@app.route('/')
def hello():
    return render_template('index.html', utc_dt=datetime.datetime.utcnow())


@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Check if the file is an allowed format (e.g. JPEG, PNG)
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    # Save the file to disk
    filename = secure_filename(file.filename)
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Return a JSON response with the filename and a success message
    return jsonify({'filename': filename,
                    'message': 'File uploaded successfully',
                    })
