
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
    parser.add_argument('--pred_json', type=str, default=r'', help='pred coco json path')
    parser.add_argument('--model_path', type=str, default=r'/userhome/lhf/Github/WorkSpace/VisDrone/ultralytics/runs/detect/yolov11n-640', help='root_path + "/weights/best.pt"')
    # project_path
    parser.add_argument('--project', type=str, default=r'/userhome/lhf/Github/WorkSpace/VisDrone/mAPs', help='project path')
    parser.add_argument('--name', type=str, default='exp', help='name for evaluation')
    # for iou
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold for evaluation')
    # for conf
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold for evaluation')
    # for imgsz
    parser.add_argument('--img_size', type=int, default=1024, help='image size for evaluation')
    # batch_size
    parser.add_argument('--batch', type=int, default=64, help='batch size for evaluation')
    # for tidecv flag
    parser.add_argument('--tidecv', action="store_true", help=' store_false or store_true whether to use tidecv for evaluation')
    # for device
    parser.add_argument('--device', type=str, default='cuda', help='device for evaluation')
    return parser

parser = parse_opt()
# parser.print_help()
opt = parser.parse_args()
anno_json = opt.anno_json
pred_json = opt.pred_json
model_path = opt.model_path
project_path = opt.projectw

# log iou_thres and conf_thres
print_log(f">>> iou_thres: {opt.iou_thres} . <<<+")
print_log(f">>> conf_thres: {opt.conf_thres} . <<<+")
# check is root_path is exist
if not os.path.exists(Path(model_path)):
    print_log(f">>> root_path not exists.  model_path: {model_path} <<<+",'yellow')
    exit()
else:
    if not os.path.exists(model_path + "/weights/best.pt"):
        print_log(">>> best.pt not exists. <<<+",'yellow')
    print_log(f">>> model path: {model_path} . <<<+")
    
# check if project_path exists if not create it
if not os.path.exists(project_path):
    os.makedirs(project_path)
    print_log(f">>> create project_path *{project_path}* . <<<+",'yellow')
else:
    print_log(f">>> project_path *{project_path}* exists. <<<+")

# from ultralytics import settings

# # update datasets_dir weights_dir and other all settings
# settings.update({
#     'weights_dir': '/userhome/lhf/Github/WorkSpace/office_weights',
#     'runs_dir': root_path,
#     'tensorboard': False,
#     'wandb': False,
# })

from ultralytics.models.yolo.detect import DetectionValidator

from ultralytics.models.rtdetr import RTDETRValidator

args = {
    "model": model_path + "/weights/best.pt", 
    "data": "/userhome/lhf/Github/WorkSpace/ultralytics_cfg/cfg/datasets/VisDrone.yaml", 
    "save_json": True, 
    "imgsz": opt.img_size,
    "mode": "test", #保存文件夹名称
    "split": "test", # 控制读取哪个split的数据集
    "project": project_path + "/get_map", # 保存文件夹名称
    "name": opt.name, # 保存文件夹名称
    "batch": opt.batch,
    "iou": 0.5,
    "conf": 0.25, 
    "device":opt.device,
    }

if pred_json == '':
    print_log(">>> No pred_json provided, will validate once to get pred_annotations. <<<+",'yellow')
    ## ==========> run ultralytics validation ===========
    print_log("============================>>> Start validating YOLO model predictions on COCO dataset using pycocotools. <<<=")
    if 'rtdetr' in model_path:
        validator = RTDETRValidator(args=args)
    else:
        validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    _,pred_json = validator.eval_json_polaris(validator.stats,anno_json_path=anno_json)
# else:
print_log(f">>> pred_json provided, will validate with {pred_json}. <<<+")
from pycocotools.coco import COCO  # noqa
from pycocotools.cocoeval import COCOeval  # noqa

anno = COCO(str(anno_json))  # init annotations api
pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
val = COCOeval(anno, pred, "bbox", areaRng_subset=True, infer_size = [opt.img_size, opt.img_size])
# val.params.imgIds = [str(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
val.mAP_type = 'YOLO'
print_log('evaluate starting...')
val.evaluate()
print_log('accumulate starting...')
val.accumulate()
print_log(f'summarize starting... iou_thres={opt.iou_thres}, conf_thres={opt.conf_thres} infer_size={opt.img_size} mAP_type:{val.mAP_type}')
val.summarize(TOD=True)
print_log(f'==========>>> evaluate done for model_path:{model_path} ...')

if opt.tidecv:
    print_log("==========>>> Start validating YOLO model predictions on COCO dataset using tidecv. <<<=")
    from tidecv import TIDE, datasets
    ## ==========> run tidecv validation =
    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    out_dir=project_path+r'/tidecv_result/'
    tide.plot(out_dir=out_dir)
    print_log(f"==========>>> tidecv validation done. out_dir{out_dir} <<<+")