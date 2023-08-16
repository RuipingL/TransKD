# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
from pickle import FALSE
import random
import time
import numpy as np
import torch
import math
# import utils #visdom
import datetime
from PIL import Image, ImageOps
from argparse import ArgumentParser
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler, AdamW
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torch_poly_lr_decay import PolynomialLRDecay
from utils import netParams
from models.Segformer4EmbeddingKD import mit_b0,mit_b1,mit_b2#,mit_b3,mit_b4,mit_b5
from CSF_EA import build_kd_trans, hcl, CriterionCWD

from dataset import VOC12,cityscapes, ACDC
from transform import Relabel, ToLabel, Colorize
import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile
NUM_CHANNELS = 3
NUM_CLASSES = 20 

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

NOW = datetime.datetime.now()
TODAY = f'{NOW.year}-{NOW.month}-{NOW.day}'

class MyCoTransform(object):
    def __init__(self, augment=True, height=512, model='SegformerB0'):
        # self.enc=enc
        self.augment = augment
        self.height = height
        self.model = model
        pass
    def __call__(self, input, target):
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


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)



def train(args, model, teacher):
    best_acc = 0


    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.5959737
    weight[1] = 6.741505
    weight[2] = 3.5353868
    weight[3] = 9.866315
    weight[4] = 9.690922
    weight[5] = 9.369371
    weight[6] = 10.289124 
    weight[7] = 9.953209
    weight[8] = 4.3098087
    weight[9] = 9.490392
    weight[10] = 7.674411
    weight[11] = 9.396925	
    weight[12] = 10.347794 	
    weight[13] = 6.3928986
    weight[14] = 10.226673 	
    weight[15] = 10.241072	
    weight[16] = 10.28059
    weight[17] = 10.396977
    weight[18] = 10.05567	

    weight[19] = 0

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(augment=True, height=args.height, model=args.model)#1024)
    co_transform_val = MyCoTransform(augment=False, height=args.height, model=args.model)#1024)
    if args.dataset == 'cityscapes':
        dataset_train = cityscapes(args.datadir, co_transform, 'train')
        dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    elif args.dataset == 'ACDC':
        dataset_train = ACDC(args.datadir, co_transform, 'train')
        dataset_val = ACDC(args.datadir, co_transform_val, 'val')
    else:
        assert 'Dataset does not exist'

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        weight = weight.to(args.device)
    criterion = CrossEntropyLoss2d(weight).to(args.device)
    criterion_cwd = CriterionCWD(args.norm_type,args.divergence,args.temperature).to(args.device)
    print(type(criterion))

    if args.embed==5:
        embedd = '-embed'
    elif args.embed ==0:
        embedd = ''
    else:
        embedd = f'-embed{args.embed}'
    if args.dataset == 'cityscapes':
        savedir = f'../save/Testbatch{args.batch_size}/Optimization_PEA/{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}{embedd}-{args.embed_weights}-initialize'
    elif args.dataset == 'ACDC':
        savedir = f'../save/Testbatch{args.batch_size}-ACDC/Optimization_PEA/{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}{embedd}-{args.embed_weights}-initialize'
    else:
        assert 'Dataset does not exist'

    if args.student_pretrained:
        pretrained = 'pretrained'
    else:
        pretrained = 'nonpretrained'

    if args.savedate:
        savefile = f'{args.model}-{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}-{pretrained}-{TODAY}'
    else:    
        savefile = f'{args.model}-{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}-{pretrained}'
    automated_log_path = savedir + f"/automated_log_{savefile}.txt"
    modeltxtpath = savedir + f"/model_{savefile}.txt"   
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    if args.model.startswith('Segformer'):
        optimizer = AdamW(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=0.01)
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=1500, end_learning_rate=0.0, power=1.0)
    else: 
        assert 'model not supported'
    start_epoch = 1

    if args.visualize and args.steps_plot > 0:
        from visualize import Dashboard 
        board = Dashboard(args.port)
    if args.teacher_val:
        print("----- TEACHER VALIDATING-----")
        teacher.eval()
        epoch_loss_val = []
        time_val = []
        doIouVal =  args.iouVal  
        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        with torch.no_grad():
            for step, (images, labels) in enumerate(loader_val):
                start_time = time.time()
                if args.cuda:
                    images = images.to(args.device)
                    labels = labels.to(args.device)

                inputs = Variable(images)   
                targets = Variable(labels)
                _,outputs,_ = teacher(inputs) 

                loss = criterion(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())
                time_val.append(time.time() - start_time)


                if (doIouVal):
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print(f'VAL loss: {average:0.4} ( step: {step})', 
                                "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

            iouVal = 0
            if (doIouVal):
                iouVal, iou_classes = iouEvalVal.getIoU()
                iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
                print ("EPOCH IoU on VAL set: ", iouStr, "%") 

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        teacher.eval()
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()

            inputs = Variable(images).to(args.device)
            targets = Variable(labels).to(args.device)
            if args.embed:
                fstudent,outputs, estudent = model(inputs)
                fteacher,_,eteacher = teacher(inputs)
            else:
                fstudent,outputs = model(inputs)
                fteacher,_,_ = teacher(inputs)
            loss_logits = criterion(outputs, targets[:, 0])
            optimizer.zero_grad()
            if args.knowledge_distillation_loss == 'hcl':
                loss_review = hcl(fstudent, fteacher)*args.review_kd_loss_weight
                loss = loss_review+ loss_logits
            elif args.knowledge_distillation_loss == 'CWloss':
                loss_cwd = 0.0
                for i in range(len(fstudent)):
                    loss_cwd += criterion_cwd(fstudent[i],fteacher[i])
                loss = args.lambda_cwd*loss_cwd+ loss_logits
            if args.embed!=0:
                loss_embedd = nn.MSELoss()
                if args.embed == 5:
                    embed_weights= list(args.embed_weights)
                    for i in range(len(estudent)):
                        loss += embed_weights[i]*loss_embedd(estudent[i], eteacher[i])
                else:
                    loss += loss_embedd(estudent[0], eteacher[args.embed-1])      

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if args.visualize and args.visualize_outimages and args.steps_plot > 0 and step % args.steps_plot == 0 :
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'input step: {step}')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output  step: {step}')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output  step: {step}')
                board.image(color_transform(targets[0].cpu().data),
                    f'target step: {step}')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        scheduler.step()    
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        with torch.no_grad():
            for step, (images, labels) in enumerate(loader_val):
                start_time = time.time()
                if args.cuda:
                    images = images.to(args.device)
                    labels = labels.to(args.device)

                inputs = Variable(images)    
                targets = Variable(labels)
                if args.embed!=0:
                    _,outputs,_ = model(inputs) 
                else:
                    _,outputs = model(inputs) 

                loss = criterion(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())
                time_val.append(time.time() - start_time)


                if (doIouVal):
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                if args.visualize and args.visualize_outimages and args.steps_plot > 0 and step % args.steps_plot == 0:
                    start_time_plot = time.time()
                    image = inputs[0].cpu().data
                    board.image(image, f'VAL input step: {step})')
                    if isinstance(outputs, list):   #merge gpu tensors
                        board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                        f'VAL output step: {step}')
                    else:
                        board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                        f'VAL output , step: {step}')
                    board.image(color_transform(targets[0].cpu().data),
                        f'VAL target step: {step})')
                    print ("Time to paint images: ", time.time() - start_time_plot)
                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                            "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
            if args.visualize:
                board.add_scalar(win=f'Validation IoU {args.model} {pretrained} {args.knowledge_distillation_feature_fusion} {args.knowledge_distillation_loss}', x=epoch, y=iouVal)
           

        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        
        filenameCheckpoint = savedir + f'/checkpoint_{savefile}.pth.tar'
        filenameBest = savedir + f'/model_best_{savefile}.pth.tar'    

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)


        filename = f'{savedir}/model_{savefile}-{epoch:03}.pth'
        filenamebest = f'{savedir}/model_{savefile}_best.pth'

        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            with open(savedir + f"/best_{savefile}.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))
                myfile.write(class_iou_messages(iou_classes))   
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
        if args.visualize:    
            board.add_doubleline(epoch=epoch, val_loss=average_epoch_loss_val, train_loss=average_epoch_loss_train, title=f'Losses {pretrained} {args.knowledge_distillation_feature_fusion} {args.knowledge_distillation_loss}', win= f'Losses {args.model}')
   
    return(model)  

def class_iou_messages(iou_classes):
    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)
    classes = ["Road","sidewalk","building","wall","fence","pole","traffic light","traffic sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"]
    print_iou_classes = f'\nPer-Class IoU:'
    for i in range(iou_classes.size(0)):
        iou = iou_classes_str[i]
        classi = classes[i]
        print_iou_classes += f'\n{iou}\t{classi} '
    return print_iou_classes

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

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
    if args.embed==5:
        embedd = '-embed'
    elif args.embed ==0:
        embedd = ''
    else:
        embedd = f'-embed{args.embed}'

    if args.dataset == 'cityscapes':
        savedir = f'../save/Testbatch{args.batch_size}/Optimization_PEA/{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}{embedd}-{args.embed_weights}-initialize'
    elif args.dataset == 'ACDC':
        savedir = f'../save/Testbatch{args.batch_size}-ACDC/Optimization_PEA/{args.knowledge_distillation_feature_fusion}-{args.knowledge_distillation_loss}{embedd}-{args.embed_weights}-initialize'
    else:
        assert 'Dataset does not exist'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    teacher = mit_b2()
    if args.dataset == 'cityscapes':
        teacher_path = '../outputs/segformerb2_teacher_cityscapes.pth'
    elif args.dataset == 'ACDC':
        teacher_path = '../outputs/segformerb2_teacher_acdc.pth'
    else:
        assert 'dataset not supported'

    if args.model.startswith('Segformer'):
        if args.model.endswith('B0'):
            model = mit_b0()
            path = 'ckpt_pretrained/mit_b0.pth'
        elif args.model.endswith('B1'):
            model = mit_b1()
            path = 'ckpt_pretrained/mit_b1.pth'  
        if args.student_pretrained:   
            print('load weights from pretrained ckpt ...')               
            save_model = torch.load(path)
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            print('no pretrained ckpt loaded ...')
    else: 
        assert 'unsupported model'

    model = build_kd_trans(model, args.embed)
    if args.continue_path is not None:
        continue_path = args.continue_path
        model = load_my_state_dict(model, torch.load(continue_path),False)
        print('load continue path:', continue_path)

    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
    if args.cuda:
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            teacher = load_my_state_dict(teacher,torch.load(teacher_path))#True
            model=model.to(args.device)
            teacher = teacher.to(args.device)
            model = nn.DataParallel(model)  
            teacher = nn.DataParallel(teacher)
        else:
            print("single GPU for training")
            teacher = load_my_state_dict(teacher,torch.load(teacher_path))
            model = model.to(args.device) 
            teacher = teacher.to(args.device)
    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder
    
    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """


    print("========== STUDENT TRAINING ===========")
    model = train(args, model, teacher) 

    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="SegformerB0")

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--dataset',default="cityscapes", choices=['ACDC','cityscapes'])
    parser.add_argument('--datadir', default="/path/to/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    
    parser.add_argument('--savedate', default=True)
    parser.add_argument('--visualize', default=False)
    parser.add_argument('--visualize_outimages', action='store_true',default=False)
    parser.add_argument('--iouTrain', action='store_true', default=False) 
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument("--device", default='cuda', help="Device on which the network will be trained. Default: cuda")
    parser.add_argument("--knowledge-distillation-feature-fusion", default= 'CSF', choices=['CSF'])
    parser.add_argument("--knowledge-distillation-loss", default="hcl", choices= ['hcl', 'CWloss'])
    parser.add_argument('--kd-tau', default=1., type=float, help="")
    parser.add_argument('--kd-alpha', default=0.5, type=float, help="")
    parser.add_argument('--student-pretrained',default= False)
    parser.add_argument('--review-kd-loss-weight', type=float, default=1.0,
                    help='feature konwledge distillation loss weight')
    parser.add_argument('--teacher-val', default= False, help='validate teacher before training process')
    parser.add_argument("--temperature", type=float, default=1.0, help="normalize temperature")
    parser.add_argument("--lambda-cwd", type=float, default=1.0, help="lambda_kd")
    parser.add_argument("--norm-type", type=str, default='channel',choices=['channel','spatial','channel_mean'], help="kd normalize setting")#不改
    parser.add_argument("--divergence", type=str, default='kl', help="kd divergence setting")
    parser.add_argument("--embed",type=int,default=5, choices=[0,1,2,3,4,5])
    parser.add_argument('--continue-path',default=None)
    parser.add_argument('--lr',type=float,default=6e-5)
    parser.add_argument('--embed-weights',type=tuple,default=(0.1,0.1,0.5,1))


    main(parser.parse_args())
