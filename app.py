from flask import Flask, Response, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime
from models import db, VehicleCount
from utils import predict_peak_hour

app = Flask(__name__)
def strftime_filter(value, format='%Y-%m-%d'):
    if value == 'now':
        return datetime.now().strftime(format)
    return value  # Nếu không phải 'now', trả về nguyên giá trị

# Đăng ký filter với Jinja2
app.jinja_env.filters['strftime'] = strftime_filter
# Cấu hình database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:admin@127.0.0.1/nhandienxe_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Load YOLO model
model_path = r"D:\Testfiletrain\Results\yolov8_vehicle\weights\best.pt"
model = YOLO(model_path)

# Đọc video
video_path = r"D:\video\vid5.mp4"
cap = cv2.VideoCapture(video_path)

# Định nghĩa ROI và lines
roi_points = [(271, 267), (604, 225), (1135, 461), (374, 658)]
roi_points1 = [(459, 161), (861, 338), (1078, 223), (571, 103)]
roi_points2 = [(7, 217), (216, 208), (351, 680), (3, 686)]
lines = [
    [(330, 471), (867, 369)], [(475, 174), (731, 174)], [(172, 235), (246, 656)]
]

# Biến toàn cục
vehicle_counts = defaultdict(lambda: defaultdict(int))
counted_vehicles = set()
track_history = defaultdict(lambda: deque(maxlen=10))

def is_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    return cv2.pointPolygonTest(np.array(roi, np.int32), (center_x, center_y), False) >= 0

def has_crossed_line(track_id, line, current_center):
    if track_id not in track_history or len(track_history[track_id]) < 2:
        return False
    prev_center = track_history[track_id][-2]
    curr_center = current_center
    line_start, line_end = line
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return cross_product(line_start, line_end, prev_center) * cross_product(line_start, line_end, curr_center) < 0

def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        im0 = frame.copy()
        for roi, color in zip([roi_points, roi_points1, roi_points2], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            cv2.polylines(im0, [np.array(roi, np.int32)], isClosed=True, color=color, thickness=3)
        for line in lines:
            cv2.line(im0, line[0], line[1], (0, 255, 255), 2)

        results = model.track(frame, persist=True, conf=0.5, iou=0.4)
        for result in results:
            for box in result.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id)
                vehicle_type = model.names[int(box.cls)]
                center = (x1 + x2) // 2, (y1 + y2) // 2
                track_history[track_id].append(center)

                for i, (roi, line) in enumerate(zip([roi_points, roi_points1, roi_points2], lines)):
                    if is_inside_roi((x1, y1, x2, y2), roi):
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(im0, f"{vehicle_type} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if has_crossed_line(track_id, line, center) and track_id not in counted_vehicles:
                            vehicle_counts[i][vehicle_type] += 1
                            counted_vehicles.add(track_id)
                            with app.app_context():
                                new_count = VehicleCount(roi_id=i, vehicle_type=vehicle_type, count=1, timestamp=datetime.now())
                                db.session.add(new_count)
                                db.session.commit()

        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    current_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html', counts=vehicle_counts, current_date=current_date)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/report/<int:roi_id>', methods=['GET'])
def report(roi_id):
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    with app.app_context():
        counts = db.session.query(VehicleCount.vehicle_type, db.func.sum(VehicleCount.count))\
            .filter(VehicleCount.roi_id == roi_id, db.func.date(VehicleCount.timestamp) == date)\
            .group_by(VehicleCount.vehicle_type).all()
        return render_template('report.html', roi_id=roi_id, date=date, counts=dict(counts))
@app.route('/prediction/<int:roi_id>')
def prediction(roi_id):
    with app.app_context():
        peak_hour = predict_peak_hour(roi_id)
        current_date = datetime.now().strftime('%Y-%m-%d')  # Đảm bảo có current_date
        return render_template('prediction.html', roi_id=roi_id, peak_hour=peak_hour, current_date=current_date)
@app.route('/api/counts/realtime')
def get_realtime_counts():
    return jsonify({i: dict(counts) for i, counts in vehicle_counts.items()})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Tạo bảng nếu chưa có
    app.run(debug=True, host='0.0.0.0', port=5000)