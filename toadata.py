import random
from datetime import datetime, timedelta

# Danh sách loại xe mới
vehicle_types = ["xe_may", "xe_tai", "o_to", "xe_bus"]

# Hàm tạo số lượng xe theo giờ
def generate_vehicle_count(hour, vehicle_type, roi_id):
    base_count = 0
    if 7 <= hour <= 9 or 16 <= hour <= 18:  # Cao điểm
        base_count = random.randint(10, 50)
    elif 0 <= hour <= 5:  # Thấp điểm
        base_count = random.randint(0, 10)
    else:  # Bình thường
        base_count = random.randint(5, 25)

    # Điều chỉnh theo loại xe và ROI
    if roi_id == 0:  # Đông đúc
        if vehicle_type == "o_to": return base_count * 2
        if vehicle_type == "xe_may": return base_count * 1.5
    elif roi_id == 1:  # Công nghiệp
        if vehicle_type == "xe_tai": return base_count * 2
        if vehicle_type == "xe_bus": return base_count * 1.5
    else:  # Ngoại ô
        if vehicle_type == "xe_may": return base_count * 2
        if vehicle_type == "o_to": return base_count * 0.5
    return base_count

# Tạo dữ liệu
start_date = datetime(2025, 2, 5)
days = 30
sql_statements = []

for day in range(days):
    current_date = start_date + timedelta(days=day)
    for hour in range(24):
        timestamp = current_date.replace(hour=hour, minute=0, second=0)
        for roi_id in range(3):
            for vehicle_type in vehicle_types:
                count = generate_vehicle_count(hour, vehicle_type, roi_id)
                sql = f"INSERT INTO vehicle_counts (roi_id, vehicle_type, count, timestamp) " \
                      f"VALUES ({roi_id}, '{vehicle_type}', {count}, '{timestamp.strftime('%Y-%m-%d %H:%M:%S')}');"
                sql_statements.append(sql)

# Ghi vào file
with open("insert_data_full.sql", "w", encoding="utf-8") as f:
    f.write("USE traffic_db;\n")
    f.write("DELETE FROM vehicle_counts;\n")  # Xóa dữ liệu cũ
    for statement in sql_statements:
        f.write(statement + "\n")

print(f"Đã tạo {len(sql_statements)} bản ghi vào file insert_data_full.sql")