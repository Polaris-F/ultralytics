# update ultralytics settinngs
# 添加ultralytics路径
import os
import sys
sys.path.append('/userhome/lhf/Github/ultralytics')

from ultralytics import settings

# update datasets_dir weights_dir and other all settings
settings.update({
    # 'datasets_dir': '/path/to/datasets', 
    'weights_dir': '/userhome/lhf/Github/WorkSpace/office_weights',
    'runs_dir': '/userhome/lhf/Github/ultralytics/workspace/Debug_HazyData/runs',
    'tensorboard': False,
    'wandb': False,
})

from ultralytics.models.decodet import DeCoDetModel

# 确保输出目录存在
os.makedirs('/userhome/lhf/Github/ultralytics/workspace/Debug_HazyData/runs', exist_ok=True)

# 数据集配置路径
data_yaml = "/userhome/lhf/Github/ultralytics/workspace/Debug_HazyData/Hazy_Data.yaml"
if not os.path.exists(data_yaml):
    data_yaml = "Hazy_Data.yaml"  # 尝试在当前目录查找

# 加载模型
model = DeCoDetModel("/userhome/lhf/Github/ultralytics/ultralytics/cfg/models/yolov8-decodet.yaml")

# 训练模型 - 不需要传递额外的深度损失参数，已在DeCoDetLoss类中设置默认值
model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    device="0",
    batch=4,
    name="Hazy_Data_Debug",
    exist_ok=True
)