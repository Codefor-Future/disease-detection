import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory,send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
import pickle
from skimage import io, feature
from scipy.stats import entropy

app = Flask(__name__, static_folder='public')

# test
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
    # gaussian = cv2.GaussianBlur(img, (3,3), 0)
    cv2.imwrite('public/images/input.png', img)
    # # buffer = io.BytesIO()
    # cv2.imwrite('public/images/step1output.png', gaussian)
    
    gaussian = cv2.GaussianBlur(img, (3,3), 0)
    cv2.imwrite('public/images/gaussian.png', gaussian)
    # Load the input image
    #path = "bs_gaussian.png"
    #img = cv2.imread(path, 1)

    # Convert the image to HSI color space
    hsi = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)
    lower_hsi = np.array([0, 40, 0]) # Lower threshold for H (hue), S (saturation), I (intensity)
    upper_hsi = np.array([80, 255, 255]) # Upper threshold for H (hue), S (saturation), I (intensity)

    # Create a binary mask using the threshold values to segment the image
    mask = cv2.inRange(hsi, lower_hsi, upper_hsi)

    # Apply the mask to the input image to obtain the leaf image without the background
    leaf_img = cv2.bitwise_and(gaussian, gaussian, mask=mask)
    cv2.imwrite('public/images/mask.png', leaf_img)

    # cv2.imshow('Leaf Image', leaf_img)

    # Convert the image to HSI color space
    hsi = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)

    # Set the thresholds for the infected region in HSI color space
    lower_hsv = np.array([10, 50, 50])
    upper_hsv = np.array([20, 255, 255])

    # Create a mask for the infected region
    mask_infected = cv2.inRange(hsi, lower_hsv, upper_hsv)

    # Apply the mask to the input image to obtain the leaf image without the background
    leaf_img = cv2.bitwise_and(leaf_img, leaf_img, mask=mask_infected)
    cv2.imwrite('public/images/segmentation.png', leaf_img)

    # cv2.imshow('leaf image',leaf_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #feature_extraction
    # Split the color image into its three color channels
    red_channel = leaf_img[:, :, 0]
    green_channel = leaf_img[:, :, 1]
    blue_channel = leaf_img[:, :, 2]

    # Extract texture features using GLCM for each channel
    glcm_red = feature.graycomatrix(red_channel, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=False, normed=True)
    glcm_green = feature.graycomatrix(green_channel, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=False, normed=True)
    glcm_blue = feature.graycomatrix(blue_channel, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=False, normed=True)
            
    area = np.count_nonzero(leaf_img)

            # Calculate the mean, standard deviation, variance, homogeneity, entropy, and contrast for each channel
    mean_red = (feature.graycoprops(glcm_red, 'contrast')[0, 0] + feature.graycoprops(glcm_red, 'homogeneity')[0, 0]) / 2
    mean_green = (feature.graycoprops(glcm_green, 'contrast')[0, 0] + feature.graycoprops(glcm_green, 'homogeneity')[0, 0]) / 2
    mean_blue = (feature.graycoprops(glcm_blue, 'contrast')[0, 0] + feature.graycoprops(glcm_blue, 'homogeneity')[0, 0]) / 2

    std_red = np.std(leaf_img)
    std_green = np.std(leaf_img)
    std_blue = np.std(leaf_img)

    variance_red = np.var(leaf_img)
    variance_green = np.var(leaf_img)
    variance_blue = np.var(leaf_img)

    homogeneity_red = feature.graycoprops(glcm_red, 'homogeneity')[0, 0]
    homogeneity_green = feature.graycoprops(glcm_green, 'homogeneity')[0, 0]
    homogeneity_blue = feature.graycoprops(glcm_blue, 'homogeneity')[0, 0]

    entropy_red = entropy(glcm_red.ravel())
    entropy_green = entropy(glcm_green.ravel())
    entropy_blue = entropy(glcm_blue.ravel())
    contrast_red= feature.graycoprops(glcm_red, 'contrast')[0, 0]
    contrast_green= feature.graycoprops(glcm_green, 'contrast')[0, 0]
    contrast_blue= feature.graycoprops(glcm_blue, 'contrast')[0, 0]
    features_list = np.zeros((19,1))
    features_list=[area,mean_red,mean_green,mean_blue,std_red,std_green,std_blue,variance_red,variance_green,variance_blue,homogeneity_red, homogeneity_green,
                    homogeneity_blue,entropy_red,entropy_green,entropy_blue, contrast_red,contrast_green,contrast_blue]
            
    #print(features_list)

    #classification using Neural Network
    # from tensorflow.keras.models import load_model

    from tensorflow import keras
    from keras.layers import Dense
    from keras.models import Sequential, load_model

    # Load the saved model
    model = load_model('nn_model_d4.h5')
    classes=['Brownspot','Healthy','LeafBlast']
    features_list=np.array(features_list)
    features_list_reshaped = features_list.reshape(1, 19, 1)
    # Predict the class label for the new data using the trained neural network model
    class_probabilities = model.predict(features_list_reshaped)
    # Get the index of the class with highest probability
    predicted_class_nn = np.argmax(class_probabilities)

    # Print the predicted class label
    print("Predicted class label using Neural Network:", classes[predicted_class_nn])
    print('Class probability: ', round(class_probabilities[0][predicted_class_nn]*100,2), '%')

    #classification using svm
    import joblib
    # Load the SVM model from file
    clf = joblib.load('svm_model_d4.pkl')
    features_list_reshaped = features_list_reshaped.reshape(1, 19)
    predicted_class = clf.predict(features_list_reshaped)

    # Print the predicted class label
    print("Predicted class label using svm:", [classes[label] for label in predicted_class])
    data = {'svm': [classes[label] for label in predicted_class], "nn":classes[predicted_class_nn], 'nnPro':round(class_probabilities[0][predicted_class_nn]*100,2)}
    return jsonify(data)

@app.route('/')
def hello():
    return render_template('index.html', utc_dt=datetime.datetime.utcnow())

@app.route('/result')
def result():
    return render_template('result.html', utc_dt=datetime.datetime.utcnow())


# @app.route('/upload', methods=['POST'])
# def upload_image():
#     # Check if the post request has the file part
#     if 'image' not in request.files:
#         return jsonify({'error': 'No file uploaded'})

#     file = request.files['image']

#     # If user does not select file, browser also
#     # submit an empty part without filename
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'})

#     # Check if the file is an allowed format (e.g. JPEG, PNG)
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file format'})

#     # Save the file to disk
#     filename = secure_filename(file.filename)
#     # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#     # Return a JSON response with the filename and a success message
#     return jsonify({'filename': filename,
#                     'message': 'File uploaded successfully',
#                     })
