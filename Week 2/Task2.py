from yolov5 import detect

detect.run(
    weights = "yolov5x.pt",
    source=1,
    conf_thres=0.6,
    line_thickness=2,  
)