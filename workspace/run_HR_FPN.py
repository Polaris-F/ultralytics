# update ultralytics settinngs
from ultralytics import settings

# update datasets_dir weights_dir and other all settings
settings.update({
    # 'datasets_dir': '/path/to/datasets', 
    'weights_dir': '/userhome/lhf/Github/WorkSpace/office_weights',
    'runs_dir': '/userhome/lhf/Github/WorkSpace/VisDrone/ultralytics/HRFPN',
    'tensorboard': False,
    'wandb': False,
})

from ultralytics import YOLO

# Load a model
model = YOLO("HR_FPN_Detect.yaml").load('yolov5x.pt')  # build a new model from YAML``
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
model.train(
    data="VisDrone.yaml", 
    epochs=100, 
    imgsz=1024,
    device="4,5,6,7",
    batch= 8,
    name="HRFPN-Debug",
    exist_ok=True,
    )