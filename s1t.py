from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the uploaded file from the request
    file = request.files['file']
    
    # Read the image from the file
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gaussian = cv2.GaussianBlur(img, (3,3), 0)
    
    # Process the image using the code you provided
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(closing)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    masked_img = cv2.bitwise_and(img, img, mask=mask_resized)
    
    # Save the resulting image to a buffer
    buffer = io.BytesIO()
    cv2.imwrite('test.png', gaussian)
    
    # Reset the buffer to the beginning and return the file for download
    buffer.seek(0)
    return send_file('test.png', as_attachment=True)

if __name__ == '__main__':
    app.run()
