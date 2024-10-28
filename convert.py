import pandas as pd
import cv2
import os

file_path = './dataset/location.txt'

# Đọc dữ liệu từ file .txt
data = pd.read_csv(file_path, header=None)


# Kích thước ảnh mới (giả sử bạn muốn resize về 640x480)
new_width = 640
new_height = 640

for row in data.itertuples(index=False):
    img = cv2.imread(f"dataset/{row[0]}")

    # Kiểm tra nếu ảnh được đọc thành công
    if img is not None:
        orig_height, orig_width = img.shape[:2]

        # Thay đổi kích thước ảnh
        resized_img = cv2.resize(img, (new_width, new_height))

        # Lưu ảnh đã thay đổi kích thước
        cv2.imwrite(f'../NhanDangBienSo/datasets/resized/images/train/{row[0]}', resized_img)

        # Đọc nhãn và điều chỉnh theo tỷ lệ kích thước mới
        x = row[2]
        y = row[3]
        w = row[4]
        h = row[5]

        # Tính toán tọa độ trung tâm và kích thước (giá trị gốc)
        x_center_orig = (x + w / 2) / orig_width
        y_center_orig = (y + h / 2) / orig_height
        w_orig = w / orig_width
        h_orig = h / orig_height

        # Tính toán lại nhãn cho kích thước ảnh mới
        x_center_resized = x_center_orig * new_width
        y_center_resized = y_center_orig * new_height
        w_resized = w_orig * new_width
        h_resized = h_orig * new_height

        # Tạo tên file và ghi dữ liệu vào file nhãn mới
        filename = row[0][:-4]  # Lấy tên file không có phần mở rộng
        data_line = f'{row[1]} {x_center_resized / new_width} {y_center_resized / new_height} {w_resized / new_width} {h_resized / new_height}\n'

        # Ghi đè nội dung vào file nhãn mới
        with open(f'../NhanDangBienSo/datasets/resized/labels/train/{filename}.txt', 'w') as file:
            file.write(data_line)

        print(f"Đã xử lý ảnh và nhãn: {row[0]}")
    else:
        print(f"Không thể đọc ảnh: {row[0]}")
