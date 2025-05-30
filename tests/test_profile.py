import warnings
warnings.filterwarnings('ignore')

import os
import sys
ultralytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ultralytics_path)

from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('/userhome/lhf/Codes/3rdparty/ultralytics/ultralytics/cfg/models/Pro/HR_FPN_Detect.yaml')
    model.info(detailed=True)
    # try:
    #     model.profile(imgsz=[640, 640])
    # except Exception as e:
    #     print(e)
    #     pass
    model.fuse()