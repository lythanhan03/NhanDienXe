from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class VehicleCount(db.Model):
    __tablename__ = 'vehicle_counts'  # Chỉ định tên bảng rõ ràng
    id = db.Column(db.Integer, primary_key=True)
    roi_id = db.Column(db.Integer, nullable=False)
    vehicle_type = db.Column(db.String(50), nullable=False)
    count = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)