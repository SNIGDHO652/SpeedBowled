from flask import Flask, render_template, request, jsonify
import cv2
import json
import math
import numpy as np
from scipy.spatial.distance import euclidean

app = Flask(__name__)

BALL_DIAMETER_CM = 0.0725  

def detect_colored_and_circular_ball(frame, color_range=None):
    if color_range:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([max(color_range[0] - 10, 0), 50, 50])
        upper_color = np.array([min(color_range[0] + 10, 180), 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center_x, center_y, radius = circle[0], circle[1], circle[2]
            return center_x, center_y, radius

    return None  


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    video_file = request.files['video']
    color_hsv = request.form.get('color')  
    color_range = None

    if color_hsv:
        color_hsv = json.loads(color_hsv)
        h = int(color_hsv[0])
        color_range = (h,) 

    video_path = "uploaded_video.mp4"
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(fps)

    speeds = []
    b = 9.8 / (123 / 3.6)
    vx = 0
    vy = 0
    vz = 0
    prev_pos = None
    prev_vel = None
    prev_acn = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        vel = (0, 0, 0)

        ball_info = detect_colored_and_circular_ball(frame, color_range)
        if ball_info:
            center_x, center_y, radius = ball_info
            real_distance = 1 / (2 * radius)
            pos = (center_x*real_distance, center_y*real_distance, real_distance)

            speed = 0
            acn_vect = (0, 0, 0)
            if prev_pos is not None:
                vx = ((pos[0] - prev_pos[0]) * fps)  # in m/s
                vy = ((pos[1] - prev_pos[1]) * fps)
                vz = ((pos[2] - prev_pos[2]) * fps)
                vel = (vx, vy, vz)
            prev_pos = pos
            if prev_vel is not None:
                acn_vect = ((vel[0] - prev_vel[0]) * fps,
                (vel[1] - prev_vel[1]) * fps,
                (vel[2] - prev_vel[2]) * fps)
            
            c = 0
            if prev_acn is not None: 
                if prev_vel is not None:
                    c = math.sqrt( max(0 , ((acn_vect[2] + b*vel[2])**2 - (prev_acn[2] + b*prev_vel[2])**2) / ((prev_acn[0] + b*prev_vel[0])**2 + (prev_acn[1] + b*prev_vel[1])**2 - (acn_vect[0] + b*vel[0])**2 - (acn_vect[1] + b*vel[1])**2 + 0.0000000000000001)))
                    l = (9.8 * 9.8) / ((c**2)*(acn_vect[0] + b*vel[0])**2 + (c**2)*(acn_vect[1] + b*vel[1])**2 + (acn_vect[2] + b*vel[2])**2)
                    k = l*c
                    speed = math.sqrt((vx*k)**2 + (vy*k)**2 + (vz*l)**2)


            speeds.append(speed * 3.6 )
            prev_vel = vel
            prev_acn = acn_vect
    
    processed_speeds = []

    for i in range(len(speeds)-2):
        speed1 = 0
        speed2 = 0
        speed3 = 0
        if speeds[i] and speeds[i] >= 0 and speeds[i] < 100000 : 
            speed1 = min(speeds[i], 170)
        if speeds[i+1] and speeds[i+1] >= 0 and speeds[i+1] < 100000 :
            speed2 = min(speeds[i+1], 170)
        if speeds[i+2] and speeds[i+2] >= 0 and speeds[i+2] < 100000 :
            speed3 = min(speeds[i+2], 170)
        processed_speeds.append((speed1 + speed2 + speed3) / 3)

    cap.release()
    return jsonify({
        "speeds": processed_speeds,
        "max": max(processed_speeds, default=0),
        "min": min(processed_speeds, default=0),
        "avg": sum(processed_speeds) / len(processed_speeds) if processed_speeds else 0
    })


if __name__ == '__main__':
    app.run(debug=True)
