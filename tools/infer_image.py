from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8x-seg.yaml")
model.load('/home/ma-user/work/FastSAM-x.pt')
# model = YOLO("/home/ma-user/work/FastSAM-x.pt")
# # Run inference on an image
results = model.predict("/home/ma-user/work/cat.jpg", imgsz=1024)  # return a list of Results objects
# # Process results list
for r in results:
    cls = r.boxes.cls.tolist()
    conf = r.boxes.conf.tolist()
    xyxy = r.boxes.xyxy.tolist()
    # result.save(filename="cat.jpg")  # save to disk
print(cls)
print(conf)
print(xyxy)