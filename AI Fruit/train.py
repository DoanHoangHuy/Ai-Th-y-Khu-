from ultralytics import YOLO

def main():
    print("Bắt đầu training AI.")
    
    model = YOLO("yolo11n.pt")
    
    result = model.train(
        data="data.yaml",
        epochs=25,
        imgsz=640,
        plots=True
    )
    
    print("Huấn luyện hoàn tất!  Mô hình mới đã được lưu.")
    
if __name__ == "__main__":
    main()