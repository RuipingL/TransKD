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

from dataset import ACDC, cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry
from PIL import Image, ImageOps
import random




NUM_CHANNELS = 3
NUM_CLASSES = 20
color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

class MyCoTransform(object):
    def __init__(self, augment=True, height=512, model='SegformerB0'):
        # self.enc=enc
        self.augment = augment
        self.height = height
        self.model = model
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        W,H = input.size
        if self.model.startswith('Segformer'):
            target = Resize((int(H/4+0.5),int(W/4+0.5)), Image.NEAREST)(target)
        else:
            assert 'model not supported'

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

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
def main(args):
    savedir ='./save/eval_ACDC'
    savefile = f'best_{args.distillation_type}.txt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    import sys
    PROJ_DIR = '/path/to/TransKD'
    sys.path.append(os.path.join(PROJ_DIR, 'train/'))
    if args.distillation_type == 'teacher':
        from models.Segformer import mit_b2
        model = mit_b2()
        weightspath = '../outputs/segformerb2_teacher_acdc.pth'
    elif args.distillation_type == 'student':
        from models.Segformer import mit_b0
        model = mit_b0()
        weightspath = '../outputs/segformerb0_student_acdc.pth'
    elif args.distillation_type == 'TransKDBase':
        from models.Segformer4EmbeddingKD import mit_b0
        from CSF import build_kd_trans
        model = mit_b0()
        model = build_kd_trans(model,5)    
        weightspath = '../outputs/segformerb0_TransKDBase_acdc.pth'
    elif args.distillation_type == 'TransKD_EA':
        from models.Segformer4EmbeddingKD import mit_b0
        from CSF_EA import build_kd_trans
        model = mit_b0()
        model = build_kd_trans(model,5)    
        weightspath = '../outputs/segformerb0_TransKD_EA_acdc.pth'
    elif args.distillation_type == 'TransKD_GL':
        from models.Segformer4EmbeddingKD import mit_b0
        from CSF_GLMixer import build_kd_trans
        model = mit_b0()
        model = build_kd_trans(model,5,False)    
        weightspath = '../outputs/segformerb0_TransKD_GL_acdc.pth'
    with open(savedir +'/'+ savefile, "w") as myfile:
        myfile.write(f'weight path to be evaluated: {weightspath}')
    print ("Loading weights: " + weightspath)


    #model = torch.nn.DataParallel(model)




    model = load_my_state_dict(model,torch.load(weightspath))
    print ("Model and weights LOADED successfully")
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")
    cons = ['All','fog/','night/','rain/','snow/']
    for con in cons:
        co_transform_val = MyCoTransform(augment=False, height=args.height, model=args.model)#1024)
        dataset_val = ACDC(args.datadir, co_transform_val, 'val', cons=con)
        loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


        iouEvalVal = iouEval(NUM_CLASSES)

        start = time.time()

        for step, (images, labels, filename, filenameGt) in enumerate(loader_val):
            if (not args.cpu):
                images = images.cuda()
                labels = labels.cuda()
            inputs = Variable(images)
            with torch.no_grad():
                if args.distillation_type == 'TransKDBase'or args.distillation_type == 'TransKD_EA' or args.distillation_type == 'TransKD_GL':
                    _,outputs,_ = model(inputs) 
                else:
                    outputs = model(inputs)

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)


            # print (step, filename)


        iouVal, iou_classes = iouEvalVal.getIoU()

        iou_classes_str = []
        for i in range(iou_classes.size(0)):
            iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
            iou_classes_str.append(iouStr)

        print("---------------------------------------")
        print("Adverse condition:", con)
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
        with open(savedir +'/'+ savefile, "a") as myfile:
            myfile.write("\n --------------------------------------- ")
            myfile.write(f"\n Adverse condition: {con}")
            myfile.write(f"\n Took {time.time()-start} seconds")
            myfile.write("\n Per-Class IoU:")
            myfile.write(f"\n{iou_classes_str[0]} Road")
            myfile.write(f"\n{iou_classes_str[1]} sidewalk")
            myfile.write(f"\n{iou_classes_str[2]} building")
            myfile.write(f"\n{iou_classes_str[3]} wall")
            myfile.write(f"\n{iou_classes_str[4]} fence")
            myfile.write(f"\n{iou_classes_str[5]} pole")  
            myfile.write(f"\n{iou_classes_str[6]} traffic light")
            myfile.write(f"\n{iou_classes_str[7]} traffic sign")
            myfile.write(f"\n{iou_classes_str[8]} vegetation")
            myfile.write(f"\n{iou_classes_str[9]} terrain")
            myfile.write(f"\n{iou_classes_str[10]} sky")
            myfile.write(f"\n{iou_classes_str[11]} person")
            myfile.write(f"\n{iou_classes_str[12]} rider")
            myfile.write(f"\n{iou_classes_str[13]} car")
            myfile.write(f"\n{iou_classes_str[14]} truck")
            myfile.write(f"\n{iou_classes_str[15]} bus")
            myfile.write(f"\n{iou_classes_str[16]} train")
            myfile.write(f"\n{iou_classes_str[17]} motorcycle")
            myfile.write(f"\n{iou_classes_str[18]} bicycle")
            myfile.write(f"\nMEAN IoU: {iouStr}%")



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--distillation-type', type=str, choices=['teacher', 'student','TransKDBase','TransKD_EA','TransKD_GL'])
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--model', default="SegformerB2")
    parser.add_argument('--datadir', default='/cvhci/temp/rliu/Dataset/ACDC/' )
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
