import cv2
import numpy as np
import os

from flask import Flask, request, send_file, jsonify
from io import BytesIO
from datetime import datetime
from PIL import Image
from copy import deepcopy

from src.dlx import Sudoku
from src.sudoku_picture import SudokuPicture

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = './temp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return app.send_static_file('index.html')

solver = SudokuPicture()

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    file = request.files['image']
    
    # Read image file
    image_data = file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # # Get Current Timestamp in ms - append timestamp to image name so we always save a new image
    # timestamp = datetime.now()
    # time_formatted = timestamp.strftime("%Y%m%d%H%M%S")
    # file_name = "pic_" + time_formatted + ".jpg"
    
    # Image.fromarray(image).save(f'{UPLOAD_FOLDER}/{file_name}')
    
    # Process the image (example: apply grayscale and edge detection)
    sudoku_solver = deepcopy(solver)
    result = sudoku_solver.solve(image)
    
    if type(result) == str:
        return result
    elif result is None:
        return "0"
    else:
        # Convert back to JPEG format
        is_success, buffer = cv2.imencode(".jpg", result)
        
        # Return the processed image
        return send_file(BytesIO(buffer), mimetype='image/jpeg')

@app.route('/solve_sudoku', methods=['POST'])
def solve_sudoku_state():
    
    state = request.json['sudoku']
    
    if not state:
        return jsonify({'error': 'No sudoku data provided'}), 400
    
    solver = Sudoku(state)
    result = solver.solve()
    solution = None
    for solution in result:
        break
    if solution is None:
        return jsonify({'solved': False, 'error': 'No solution exists for this sudoku.'})
    else:
        return jsonify({'solved': True, 'solution': "".join(solution)})

if __name__ == '__main__':
    # For development only - use a proper WSGI server in production
    app.run(debug=True)
    
    
