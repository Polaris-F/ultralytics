
"""Validate YOLO model predictions on COCO dataset using pycocotools."""
from datetime import datetime
import os
from pathlib import Path
import argparse





def print_log(s, content_color = 'green'):
    '''
    PrintColor = <black> <red> <green> <yellow> <blue> <amaranth>  <ultramarine> <white> \\
    PrintStyle = <default> <highlight> <underline> <flicker> <inverse> <invisible>
    '''
    # 直接转成字符串  如果输入 颜色 配置选项不在选项中
    if content_color not in ['black','red','green','yellow','blue','amaranth','ultramarine','white','']:
        s = s + '\033[0;31m x_x \033[0m' +  str(content_color)
        content_color = 'green'
    PrintColor = {'black': 30,'red': 31,'green': 32,'yellow': 33,'blue': 34,'amaranth': 35,'ultramarine': 36,'white': 37}
    PrintStyle = {'default': 0,'highlight': 1,'underline': 4,'flicker': 5,'inverse': 7,'invisible': 8}

    time_style = PrintStyle['default']
    content_style = PrintStyle['default']
    time_color = PrintColor['blue']
    content_color = PrintColor[content_color]

    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = '\033[{};{}m[{}]\033[0m \033[{};{}m{}\033[0m'.format \
        (time_style, time_color, cur_time, content_style, content_color, s)
    print (log)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=r'/userhome/lhf/datasets/VisDrone2019/VisDrone2019-DET-test-dev/data.json', help='label coco json path')
    parser.add_argument('--pred_json', type=str, default='', help='pred coco json path')
    parser.add_argument('--model_path', type=str, default=r'/userhome/lhf/Github/WorkSpace/VisDrone/ultralytics/runs/detect/yolov11n-640', help='root_path + "/weights/best.pt"')
    # for iou
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold for evaluation')
    # for conf
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold for evaluation')
    # for imgsz
    parser.add_argument('--img_size', type=int, default=640, help='image size for evaluation')
    # for tidecv flag
    parser.add_argument('--tidecv', action="store_true", help='whether to use tidecv for evaluation')

    return parser.parse_known_args()[0]

opt = parse_opt()
anno_json = opt.anno_json
pred_json = opt.pred_json
root_path = opt.model_path



# check is root_path is exist
if not os.path.exists(Path(root_path)):
    print_log(">>> root_path not exists.  *{root_path}* <<<+",'yellow')
    exit()
else:
    if not os.path.exists(root_path + "/weights/best.pt"):
        print_log(">>> best.pt not exists. <<<+",'yellow')
    print_log(f">>> model path {root_path} . <<<+")
    

# from ultralytics import settings

# # update datasets_dir weights_dir and other all settings
# settings.update({
#     'weights_dir': '/userhome/lhf/Github/WorkSpace/office_weights',
#     'runs_dir': root_path,
#     'tensorboard': False,
#     'wandb': False,
# })

from ultralytics.models.yolo.detect import DetectionValidator


args = {
    "model": root_path + "/weights/best.pt", 
    "data": "/userhome/lhf/Github/WorkSpace/ultralytics_cfg/cfg/datasets/VisDrone.yaml", 
    "save_json": True, 
    "imgsz": 640,
    "mode": "test", #保存文件夹名称
    "split": "test", # 控制读取哪个split的数据集
    "project": root_path + "/get_map", # 保存文件夹名称
    "name": "exp", # 保存文件夹名称
    "batch": 64,
    "iou": 0.5,
    "conf": 0.25, 
    }

if pred_json == '':
    print_log(">>> No pred_json provided, will validate once to get pred_annotations. <<<+",'yellow')
    ## ==========> run ultralytics validation ===========
    print_log("============================>>> Start validating YOLO model predictions on COCO dataset using pycocotools. <<<=")
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    _,pred_json = validator.eval_json_polaris(validator.stats,anno_json_path=anno_json)
else:
    print_log(f">>> pred_json provided, will validate with {pred_json}. <<<+")

if opt.tidecv:
    print_log("==========>>> Start validating YOLO model predictions on COCO dataset using tidecv. <<<=")
    from tidecv import TIDE, datasets
    ## ==========> run tidecv validation =
    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir=root_path+r'/tidecv_result/')