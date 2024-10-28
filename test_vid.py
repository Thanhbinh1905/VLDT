import numpy as np
from ultralytics import YOLO
import cv2

# Đường dẫn đến video nguồn và video đầu ra
video_path = "D:/AT Energy/test_video.mp4"
output_video_path = "D:/AT Energy/output_video.mp4"
model_path = "D:/WorkSpace/runs/detect/train24/weights/best.pt"

if __name__ == '__main__':
    # Load mô hình YOLO
    model = YOLO(model_path)

    # Mở video bằng OpenCV
    video = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có được mở thành công không
    if not video.isOpened():
        print("Không thể mở video.")
        exit()

    # Lấy các thông số của video gốc
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Định nghĩa codec và tạo đối tượng VideoWriter để ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho file mp4
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()

        # Nếu không còn khung hình nào thì thoát
        if not ret:
            break

        # Dự đoán kết quả với mỗi khung hình của video
        results = model.predict(frame)

        for r in results:
            # Vẽ kết quả của YOLO lên khung hình
            im_arr = r.plot()

        # Ghi khung hình đã xử lý vào video đầu ra
        output_video.write(im_arr)

        # Hiển thị khung hình đã xử lý (tuỳ chọn)
        cv2.imshow("YOLO Video Detection", im_arr)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng video và đóng tất cả các cửa sổ
    video.release()
    output_video.release()  # Quan trọng: cần giải phóng video đầu ra
    cv2.destroyAllWindows()

    print(f"Video đã được lưu tại: {output_video_path}")
