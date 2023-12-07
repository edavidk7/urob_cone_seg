# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import os
import time
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import torch
import argparse
import whisper
import numpy as np

from gradio import processing_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from tasks import *

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--username', metavar="FILE", help='path to data directory')
    parser.add_argument('--conf_files', default="/local/temporary/UROB/segmentation/SEEM/demo_code/configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file', )
    args = parser.parse_args()

    return args

'''
build args
'''
args = parse_option()
opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

# META DATA
cur_model = 'Focal-L'
pretrained_pth = '/local/temporary/UROB/segmentation/SEEM/seem_focall_v1.pt'
# if 'focalt' in args.conf_files:
#     pretrained_pth = os.path.join("seem_focalt_v2.pt")
#     if not os.path.exists(pretrained_pth):
#         os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
#     cur_model = 'Focal-T'
# elif 'focal' in args.conf_files:
#     pretrained_pth = os.path.join("seem_focall_v1.pt")
#     if not os.path.exists(pretrained_pth):
#         os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
#     cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)



class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)

class Video(gr.components.Video):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from xdecoder.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
import os
import glob
import subprocess
from PIL import Image
import random
from tqdm import tqdm

t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[0.99, 0.99, 0.99]]

# data_root = '/mnt/home.dokt/vacekpa2/SEEM/students/'

username = args.username

data_root = '/local/temporary/UROB/segmentation/students/' + str(username)



_ = os.stat(data_root).st_mtime  # to refresh the cache

img_files = sorted(glob.glob(data_root + '/rgb/*'))

print('found {} images'.format(len(img_files)))

with torch.no_grad():

    for idx, img_path in tqdm(enumerate(img_files)):
        # try:
        # skip already processed files
        # what about numbering
        # if os.path.exists(img_path.replace('/rgb/', '/vis/')):
            # print('skipping ', img_path, ' as it exists')
            # continue

        image = Image.open(img_path)

        image_ori = transform(image)
        # mask_ori = image['mask']
        width = image_ori.size[0]
        height = image_ori.size[1]
        image_ori = np.asarray(image_ori)
        visual = Visualizer(image_ori, metadata=metadata)
        images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

        # stroke_inimg = None
        # stroke_refimg = None

        data = {"image": images, "height": height, "width": width}

        tasks = ["Panoptic"]

        # inistalize task
        model.model.task_switch['spatial'] = False
        model.model.task_switch['visual'] = False
        model.model.task_switch['grounding'] = False
        model.model.task_switch['audio'] = False

        batch_inputs = [data]
        # if 'Panoptic' in tasks:
        model.model.metadata = metadata
        results = model.model.evaluate(batch_inputs)
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
        res = demo.get_image()
        output = Image.fromarray(res)
        segmentation = Image.fromarray(pano_seg.cpu().numpy().astype(np.uint8))

        for folder in ['seg', 'vis']:
            base_dir = os.path.dirname(img_path).replace('/rgb', '/{}'.format(folder))
            os.makedirs(base_dir, exist_ok=True)

        output.save(img_path.replace('/rgb/', '/vis/'))
        segmentation.save(img_path.replace('/rgb/', '/seg/'))


        # except:
            # print('error')

print('Annotations Finished!')