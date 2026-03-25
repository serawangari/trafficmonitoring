# training/train.py
from ultralytics import YOLO
import argparse

def train(data_yaml, epochs=50, imgsz=640, batch=16, device=0):
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,          # increase to 16 or 32 on a proper GPU
        device=device,
        workers=4,
        cache=True,           # cache dataset in RAM — fine on a real server
        amp=True,
        plots=True            # generate training plots
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.device)