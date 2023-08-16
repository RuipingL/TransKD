# Code to produce colored segmentation output in Pytorch for all cityscapes subsets  
# Sept 2017
# Eduardo Romera
#######################

from itertools import count
import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

import visdom
from mmcv.cnn import get_model_complexity_info

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024),Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize((128,256),Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

cityscapes_trainIds2labelIds = Compose([
    Relabel(19, 255),  
    Relabel(18, 33),
    Relabel(17, 32),
    Relabel(16, 31),
    Relabel(15, 28),
    Relabel(14, 27),
    Relabel(13, 26),
    Relabel(12, 25),
    Relabel(11, 24),
    Relabel(10, 23),
    Relabel(9, 22),
    Relabel(8, 21),
    Relabel(7, 20),
    Relabel(6, 19),
    Relabel(5, 17),
    Relabel(4, 13),
    Relabel(3, 12),
    Relabel(2, 11),
    Relabel(1, 8),
    Relabel(0, 7),
    Relabel(255, 0),
    ToPILImage(),
])

def main(args):
    savedir =f'./save_color_cityscapes/{args.distillation_type}'
    savefile = f'MParams_GFLOPs_{args.distillation_type}.txt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # modelpath = args.loadDir + args.loadModel
    import sys
    PROJ_DIR = '/path/to/TransKD'
    sys.path.append(os.path.join(PROJ_DIR, 'train/'))
    if args.distillation_type == 'teacher':
        from models.Segformer import mit_b2
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
    input_shape = (3,1024,2048)
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    with open(savedir +'/'+ savefile, "w") as myfile:
        myfile.write(f'weight path to be evaluated: {weightspath}')
        myfile.write('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))

    print ("Loading weights: " + weightspath)
  


    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
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
    print ("Model and weights LOADED successfully")
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()
    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if (args.visualize):
        vis = visdom.Visdom()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()

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


        label = outputs[0].max(0)[1].byte().cpu().data
        label_color = Colorize()(label.unsqueeze(0))

        filenameSave = savedir + filename[0].split("leftImg8bit/")[1]
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)   
        label_save = ToPILImage()(label_color)           
        label_save.save(filenameSave) 

        if (args.visualize):
            vis.image(label_color.numpy())
        print (step, filenameSave)

    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--subset', default="val")  
    parser.add_argument('--distillation-type',default='TransKDBase',choices=['teacher','student','TransKDBase','TransKD_GL','TransKD_EA'])

    parser.add_argument('--datadir', default="/path/to/cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
