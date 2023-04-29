import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory,send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt

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

@app.route('/process_image', methods=['POST'])
def process_image():
    print("inside")
    # Get the uploaded file from the request
    file = request.files['file']
    
    # Read the image from the file
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gaussian = cv2.GaussianBlur(img, (3,3), 0)
    
    # buffer = io.BytesIO()
    cv2.imwrite('step1output.png', gaussian)
    
    # step 2
    path = "step1output.png"
    img = cv2.imread(path, 1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to separate foreground and background
    thresh = cv2.threshold(gray, 150 , 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # Perform morphological operations to remove small noise regions
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours of the foreground objects
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a binary mask for the foreground
    mask = np.zeros_like(closing)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    # Resize the mask image to match the dimensions of the original image
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Apply the mask to the input image
    masked_img = cv2.bitwise_and(img, img, mask=mask_resized)
    cv2.imwrite('step2output.png', masked_img)

    # cv2.imshow('Masked Image', masked_img)
    
    # Reset the buffer to the beginning and return the file for download
    # buffer.seek(0)
    return send_file('step2output.png', as_attachment=True)

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
