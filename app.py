import cv2
import numpy as np
import base64
import logging
from flask import Flask, request, jsonify, render_template
from io import BytesIO
from PIL import Image
import math
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def hex_to_bgr(hex_color):
    """Convert hex color to BGR color."""
    hex_color = hex_color.lstrip('#')
    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr_color

def adaptive_color_filter(image, target_color_bgr, range_h, range_s, range_v):
    """Improved color filtering with dynamic range adjustment and histogram equalization."""
    target_hsv = cv2.cvtColor(np.uint8([[target_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    target_h, target_s, target_v = target_hsv
    lower_bound = np.array([target_h - range_h*target_h, max(0, target_s - range_s*target_s), max(0, target_v - range_v*target_v)])
    upper_bound = np.array([target_h + range_h*target_h, min(255, target_s + range_s*target_s), min(255, target_v + range_v*target_v)])
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:,2] = cv2.equalizeHist(hsv_image[:,:,2])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    cv2.imwrite('img.jpg', mask)
    return mask

def is_circle_color_valid(image, circle, target_color_bgr, color_threshold=40):
    """Verify if the specified color is within the detected circle."""
    x, y, r = circle
    mask = np.zeros_like(image[:,:,0])
    cv2.circle(mask, (x, y), r, 255, thickness=-1)

    # Extract the region of interest (ROI) containing the circle
    circle_pixels = cv2.bitwise_and(image, image, mask=mask)
    circle_pixels_hsv = cv2.cvtColor(circle_pixels, cv2.COLOR_BGR2HSV)

    # Convert target color to HSV
    target_color_hsv = cv2.cvtColor(np.uint8([[target_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # Calculate the mean color within the circle
    mean_hsv = cv2.mean(np.abs(circle_pixels_hsv), mask=mask)[:3]
    var_color = np.array(np.abs(mean_hsv - target_color_hsv))
    return var_color[0] < 41 and var_color[1] < 51 and var_color[2] < 255

def detect_ball(image, ball_color_hex):
    """Detect the ball in the image using advanced techniques."""
    ball_color_bgr = hex_to_bgr(ball_color_hex)
    mask = adaptive_color_filter(image, ball_color_bgr, range_h=.20, range_s=.3, range_v=1)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    edges = cv2.Canny(blurred, 50, 150)
    
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000, param1=100, param2=30, minRadius=1, maxRadius=2000)
    
    output_image = image.copy()
    valid_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for circle in circles:
            if is_circle_color_valid(image, circle, ball_color_bgr):
                x, y, r = circle
                cv2.circle(output_image, (x, y), r, (0, 255, 0), 4)
                valid_circles.append(circle)
        cv2.imwrite('out.jpg', output_image)
        if valid_circles:
            largest_circle = max(valid_circles, key=lambda c: c[2])
            x, y, radius = largest_circle
            return (x, y), 2 * radius
    logging.warning("No valid circles detected")
    return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_focal_length', methods=['POST'])
def calculate_focal_length():
    logging.info('Calculating focal length')
    try:
        data = request.json
        image_data = data['image']
        ball_diameter = float(data['diameter'])
        ball_color = data['color']

        # Decode the image
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect the ball in the image
        _, ball_diameter_pixels = detect_ball(image, ball_color)
        if ball_diameter_pixels is None:
            logging.warning('Ball not detected in the image')
            return jsonify({'error': 'Ball not detected'}), 400

        # Calculate the focal length
        distance_from_camera = 100  # 100 cm
        focal_length = (ball_diameter_pixels * distance_from_camera) / ball_diameter

        logging.debug(f'Focal length calculated: {focal_length}')
        return jsonify({'focal_length': focal_length})
    except Exception as e:
        logging.error(f'Error in calculating focal length: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/calculate_speeds', methods=['POST'])
def calculate_speeds():
    logging.info('Calculating speeds')
    try:
        video_file = request.files['video']
        ball_color = request.form['color']
        focal_length = float(request.form['focal_length'])
        ball_diameter = float(request.form['diameter'])
        start_time = float(request.form['start_time'])
        end_time = float(request.form['end_time'])

        video_file_path = os.path.join('uploaded_videos', video_file.filename)
        video_file.save(video_file_path)

        # Open the video file
        cap = cv2.VideoCapture(video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        time_interval = 1 / fps  # Time between frames

        positions = []
        speeds = []
        times = []
        ball_pixel_diameters = []

        frame_count = 0

        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        while cap.isOpened():
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if current_time > end_time:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Detect the ball in the current frame
            center, ball_pixel_diameter = detect_ball(frame, ball_color)
            if ball_pixel_diameter is not None:
                ball_pixel_diameters.append(ball_pixel_diameter)
            if center is not None:
                positions.append(center)
                times.append(frame_count * time_interval)  # Convert to milliseconds
            frame_count += 1
        cap.release()

        # Calculate real-world coordinates and speeds
        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]

            # Convert pixel coordinates to real-world coordinates
            Z1 = (focal_length * ball_diameter) / float(ball_pixel_diameters[i - 1])
            Z2 = (focal_length * ball_diameter) / float(ball_pixel_diameters[i])
            X1 = (x1 * Z1) / focal_length
            Y1 = (y1 * Z1) / focal_length
            X2 = (x2 * Z2) / focal_length
            Y2 = (y2 * Z2) / focal_length

            # Calculate displacement
            displacement = math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2)
            speed = (displacement / (times[i] - times[i-1])) * 0.036  # Convert cm/s to km/h
            speeds.append(speed)

        total_distance = 0

        for i in range(1, len(speeds)):
            if np.abs(speeds[i] - speeds[i - 1]) >= max(10, 0.2 * min(speeds[i], speeds[i - 1])) and max(speeds[i], speeds[i - 1]) >= 140:
                speeds[i] = min(speeds[i], speeds[i - 1])
            total_distance += ((speeds[i] + speeds[i-1])*0.5*(times[i] - times[i-1]))
        if speeds:
            min_speed = min(speeds)
            max_speed = max(speeds)
            avg_speed = 0.5*(sum(list(filter(lambda x: x <= 140, speeds))) / len(list(filter(lambda x: x <= 140, speeds))) + total_distance/(times[len(times) - 1] - times[0]))
        else:
            min_speed = max_speed = avg_speed = 0

        return jsonify({
            'min_speed': min_speed,
            'max_speed': max_speed,
            'avg_speed': avg_speed,
            'speeds': speeds,
            'times': times[:len(speeds)]
        })
    except Exception as e:
        logging.error(f'Error in calculating speeds: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploaded_videos'):
        os.makedirs('uploaded_videos')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
