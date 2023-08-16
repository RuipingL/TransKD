from typing import List
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
#################################################################################
#https://github.com/VainF/nyuv2-python-toolkit/blob/master/nyuv2.py
def colormap_NYU(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
class Colorize_NYU:

    def __init__(self, n=43):
        #self.cmap = colormap(256)
        self.cmap = colormap_NYU(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
##################################################################
# VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
#                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
#                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
#                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
#                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
#                 [0, 64, 128]]
def colormap_VOC2012(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([0, 0, 0])
    cmap[1,:] = np.array([128, 0, 0])
    cmap[2,:] = np.array([0, 128, 0])
    cmap[3,:] = np.array([128, 128, 0])
    cmap[4,:] = np.array([0, 0, 128])
    cmap[5,:] = np.array([128, 0, 128])

    cmap[6,:] = np.array([0, 128, 128])
    cmap[7,:] = np.array([128, 128, 128])
    cmap[8,:] = np.array([64, 0, 0])
    cmap[9,:] = np.array([192, 0, 0])
    cmap[10,:] = np.array([64, 128, 0])

    cmap[11,:] = np.array([192, 128, 0])
    cmap[12,:] = np.array([64, 0, 128])
    cmap[13,:] = np.array([192, 0, 128])
    cmap[14,:] = np.array([64, 128, 128])
    cmap[15,:] = np.array([192, 128, 128])

    cmap[16,:] = np.array([0, 64, 0])
    cmap[17,:] = np.array([128, 64, 0])
    cmap[18,:] = np.array([0, 192, 0])
    cmap[19,:] = np.array([128, 192, 0])
    cmap[20,:] = np.array([0 ,64, 128])
    
    return cmap
def voc_rand_crop(image, label, height, width):
    """
    Random crop image (PIL image) and label (PIL image).
    """
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(height, width))

    image = transforms.functional.crop(image, i, j, h, w)
    label = transforms.functional.crop(label, i, j, h, w)

    return image, label

class Colorize_VOC:

    def __init__(self, n=23):
        self.cmap = colormap_VOC2012(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image