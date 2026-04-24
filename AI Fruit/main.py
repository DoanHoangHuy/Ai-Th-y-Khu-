import cv2
from ultralytics import YOLO

def main():
    print("Khởi động hệ thống nhận diện trái cây & Đề xuất món ăn.")
    model = YOLO("runs/detect/train/weights/best.pt")
    print("Đã load mô hình AI thành công!")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW);
    
    if not cap.isOpened():
        print("Không mở được camera.")
        return
    print("Camera đang bật. Bấm phím 'q' để tắt!")
    # 2. Vòng lặp đọc và xử lý hình ảnh liên tục
    while True:
        # Đọc từng khung hình (frame) từ camera
        success, frame = cap.read()
        
        if success:
            # Đưa khung hình cho YOLO nhận diện
            results = model(frame)
            # Lấy hình ảnh đã được vẽ sẵn khung nhận diện (bounding box)
            annotated_frame = results[0].plot()
            # Hiển thị cửa sổ camera
            cv2.imshow("Hệ thống nhận diện AI - Nhấm 'q' để thoát.",annotated_frame)
            # Lắng nghe bàn phím, nếu bấm phím 'q' thì thoát
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        else:
            print("Không thể đọc được hình ảnh từ camera.")
            break
    # 3. Dọn dẹp: Tắt camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()