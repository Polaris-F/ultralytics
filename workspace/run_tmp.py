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


## =============>>>>>>>> 标准训练 <

if False: 
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model
    model.train(
        data="VisDrone.yaml", 
        epochs=100, 
        imgsz=640,
        device="7",
        batch=32,
        name="yolov8n-640",
        )

## =============>>>>>>>> 标准训练yolov11 640<
if True:
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    # Train the model
    model.train(
        data="VisDrone.yaml", 
        epochs=100, 
        imgsz=1024,
        device='0,1',
        batch=32,
        name="yolov11n-1024",
        exist_ok=True,
        )
