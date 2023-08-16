# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(128, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])


def main(args):

    import sys
    PROJ_DIR = '/path/to/TransKD'
    sys.path.append(os.path.join(PROJ_DIR, 'train/'))
    if args.distillation_type == 'teacher':
        from models.Segformer_ import mit_b2
        model = mit_b2()
        weightspath = '../outputs/segformerb2_teacher_cityscapes.pth'
    elif args.distillation_type == 'student':
        from models.Segformer import mit_b0
        model = mit_b0()
        weightspath = '../outputs/segformerb2_student_cityscapes.pth'
    elif args.distillation_type == 'TransKDBase':
        from models.Segformer4EmbeddingKD import mit_b0
        from CSF import build_kd_trans
        model = mit_b0()
        model = build_kd_trans(model,5)    
        weightspath='../outputs/segformerb0_TransKDBase_cityscapes.pth'
    elif args.distillation_type == 'TransKD_GL':
        from models.Segformer4EmbeddingKD import mit_b0
        from CSF_GLMixer import build_kd_trans
        model = mit_b0()
        model = build_kd_trans(model, 5, False)
        weightspath='../outputs/segformerb0_TransKD_GL_cityscapes.pth'
    elif args.distillation_type == 'TransKD_EA':
        from models.Segformer4EmbeddingKD import mit_b0
        from CSF_EA import build_kd_trans
        model = mit_b0()
        model = build_kd_trans(model, 5)
        weightspath='../outputs/segformerb0_TransKD_EA_cityscapes.pth'
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            if args.distillation_type =='teacher':
                outputs = model(inputs)
            elif args.distillation_type == 'student':
                outputs = model(inputs)
            elif args.distillation_type == 'TransKDBase':
                _,outputs,_ = model(inputs) 
            elif args.distillation_type == 'TransKD_GL':
                _,outputs,_ = model(inputs) 
            elif args.distillation_type == 'TransKD_EA':
                _,outputs,_ = model(inputs)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        # print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')
    parser.add_argument('--subset', default="val")  
    parser.add_argument('--datadir', default="/path/to/cityscapes")
    parser.add_argument('--distillation-type',default='TransKDBase',choices=['teacher','student','TransKDBase','TransKD_GL','TransKD_EA'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
