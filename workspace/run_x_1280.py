# update ultralytics settinngs
from ultralytics import settings

# update datasets_dir weights_dir and other all settings
settings.update({
    # 'datasets_dir': '/path/to/datasets', 
    'weights_dir': '/userhome/lhf/Github/WorkSpace/office_weights',
    'runs_dir': '/userhome/lhf/Github/WorkSpace/VisDrone/ultralytics/runs',
    'tensorboard': False,
    'wandb': False,
})

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.yaml")  # build a new model from YAML
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
model.train(
    data="VisDrone.yaml", 
    epochs=100, 
    imgsz=1280,
    device="4,5,6,7",
    batch=16,
    name="yolov8x-1280",
    exist_ok=True,
    )