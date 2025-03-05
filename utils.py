import numpy as np
from models import db, VehicleCount
from sqlalchemy.sql import text
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def predict_peak_hour(roi_id):
    # Lấy dữ liệu từ SQL (30 ngày)
    counts = db.session.query(db.func.hour(VehicleCount.timestamp), db.func.date(VehicleCount.timestamp), db.func.sum(VehicleCount.count)) \
        .filter(VehicleCount.roi_id == roi_id,
                VehicleCount.timestamp >= db.text("DATE_SUB(NOW(), INTERVAL 30 DAY)")) \
        .group_by(db.func.date(VehicleCount.timestamp), db.func.hour(VehicleCount.timestamp)) \
        .order_by(db.func.date(VehicleCount.timestamp), db.func.hour(VehicleCount.timestamp)) \
        .all()

    if not counts or len(counts) < 24 * 7:  # Cần ít nhất 7 ngày dữ liệu
        return {"peak_hour": "No data available", "predictions": {}}

    # Tạo mảng dữ liệu: mỗi ngày có 24 giờ
    dates = sorted(set(date for _, date, _ in counts))  # Danh sách ngày duy nhất
    hourly_data = {}
    for hour, date, count in counts:
        if date not in hourly_data:
            hourly_data[date] = [0] * 24  # Khởi tạo 24 giờ với 0
        hourly_data[date][hour] = float(count)  # Chuyển count sang float

    # Chuyển thành mảng 2D: [ngày, giờ]
    data = np.array([hourly_data[date] for date in dates])

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Chuẩn bị dữ liệu huấn luyện
    look_back = 7  # Sử dụng 7 ngày trước để dự đoán
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])  # 7 ngày trước
        y.append(scaled_data[i])  # Ngày hiện tại
    X, y = np.array(X), np.array(y)

    # Kiểm tra shape của X và y
    print("X shape:", X.shape)  # Nên là (số mẫu, 7, 24)
    print("y shape:", y.shape)  # Nên là (số mẫu, 24)

    if X.shape[0] == 0:  # Không đủ dữ liệu để tạo mẫu
        return {"peak_hour": "Insufficient data", "predictions": {}}

    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 24)))
    model.add(LSTM(50))
    model.add(Dense(24))  # Dự đoán 24 giờ
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Huấn luyện mô hình
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)

    # Dự đoán cho ngày tiếp theo
    last_sequence = scaled_data[-look_back:]  # Lấy 7 ngày cuối
    last_sequence = last_sequence.reshape((1, look_back, 24))  # Đảm bảo shape đúng
    predicted_scaled = model.predict(last_sequence)
    predicted = scaler.inverse_transform(predicted_scaled)[0]  # Chuyển về giá trị gốc

    # Tạo dictionary dự đoán cho 24 giờ
    predictions = {f"{i:02d}": round(pred, 2) for i, pred in enumerate(predicted)}
    peak_hour = int(np.argmax(predicted))  # Giờ cao điểm

    return {"peak_hour": peak_hour, "predictions": predictions}
