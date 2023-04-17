import http.server
import socketserver
import imghdr
from io import BytesIO
from PIL import Image

PORT = 8000
HTML_ENDPOINT = '/result'

class MyHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        '.html': 'text/html',
    }

class FileUploadHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == HTML_ENDPOINT:
            # Load HTML file
            with open('/home.html', 'rb') as f:
                html_data = f.read()

            # Send HTML file as response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_data)
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        file_data = self.rfile.read(content_length)

        # Validate file type
        if not self.is_valid_file_type(file_data):
            self.send_error(400, "Invalid file type")
            return

        # Open image from file data
        image_data = BytesIO(file_data)
        image = Image.open(image_data)

        # Resize image
        new_size = (image.size[0] // 2, image.size[1] // 2)
        resized_image = image.resize(new_size)

        # Save resized image to file
        with open('resized_image.jpg', 'wb') as f:
            resized_image.save(f, 'JPEG')

        # Redirect to HTML endpoint
        self.send_response(303)
        self.send_header('Location', HTML_ENDPOINT)
        self.end_headers()

    def is_valid_file_type(self, file_data):
        # Get the file type from the file data
        file_type = imghdr.what(None, file_data)

        # Only allow jpeg and png files
        return file_type in ('jpeg', 'png')

Handler = FileUploadHandler
httpd = socketserver.TCPServer(("", PORT), MyHandler)

print("Server started at localhost:{}".format(PORT))
httpd.serve_forever()
