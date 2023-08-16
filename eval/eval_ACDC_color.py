# Code to produce colored segmentation output in Pytorch for all cityscapes subsets  
# Sept 2017
# Eduardo Romera
#######################

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

from dataset import ACDC, cityscapes
from transform import Relabel, ToLabel, Colorize

import visdom
from ptflops import get_model_complexity_info
NUM_CHANNELS = 3
NUM_CLASSES = 20


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
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) 
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target
ACDC_trainIds2labelIds = Compose([
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
    savedir =f'./save_color_ACDC/{args.distillation_type}'
    savefile = f'MParams_GFLOPs_{args.distillation_type}.txt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # modelpath = args.loadDir + args.loadModel
    import sys
    PROJ_DIR = '/cvhci/temp/rliu/Projects/Distillation/KD_Framework/erfnet_old/TransKD_pytorch'
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
    macs, params = get_model_complexity_info(model, (3, 512, 912), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    with open(savedir +'/'+ savefile, "w") as myfile:
        myfile.write(f'weight path to be evaluated: {weightspath}')
        myfile.write(f'\nComputational complexity: {macs}')
        myfile.write(f"\nNumber of parameters: {params}")


    print ("Loading weights: " + weightspath)
  


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
    print ("Model and weights LOADED successfully")
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()
    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    cons = ['fog/','night/','rain/','snow/']
    for con in cons:
        co_transform_val = MyCoTransform(augment=False, height=args.height, model=args.model)#1024)
        dataset_val = ACDC(args.datadir, co_transform_val, 'val', cons=con)
        loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

        if (args.visualize):
            vis = visdom.Visdom()

        for step, (images, labels, filename, filenameGt) in enumerate(loader_val):
            if (not args.cpu):
                images = images.cuda()

            inputs = Variable(images)
            with torch.no_grad():
                if args.distillation_type =='teacher':
                    outputs = model(inputs)
                elif args.distillation_type == 'student' or args.distillation_type == 'CD':
                    outputs = model(inputs)
                elif args.distillation_type == 'TransKD_Base':
                    _,outputs,_ = model(inputs) 
                elif args.distillation_type == 'TransKD_EA' or args.distillation_type == 'TransKD_GL':
                    _,outputs,_ = model(inputs)


            label = outputs[0].max(0)[1].byte().cpu().data
            label_color = Colorize()(label.unsqueeze(0))
            file = filename[0].split('/')
            filenameSave = f'{savedir}/{con}/{args.subset}/{file[-2]}/{file[-1]}'
            os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
            label_save = ToPILImage()(label_color)           
            label_save.save(filenameSave) 

            if (args.visualize):
                vis.image(label_color.numpy())
            print (step, filenameSave)

    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--model', default="SegformerB2")
    parser.add_argument('--subset', default="val")  
    parser.add_argument('--distillation-type', type=str, choices=['teacher', 'student','TransKDBase','TransKD_EA','TransKD_GL'])

    parser.add_argument('--datadir', default="/cvhci/temp/rliu/Dataset/ACDC")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
