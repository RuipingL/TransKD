import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset
# from torchvision import transforms
import cv2

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)




class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()


        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class ACDC(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'rgb_anon/')
        self.labels_root = os.path.join(root, 'gt/')
        
        self.adverse_cons=['fog/','night/','rain/','snow/']
        self.filenames = []
        self.filenamesGt = []
        for cons in self.adverse_cons:
            images_root = self.images_root + cons + subset
            labels_root = self.labels_root + cons + subset   
            filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(images_root)) for f in fn if is_image(f)]

            filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(labels_root)) for f in fn if is_label(f)]
            self.filenames.extend(filenames)
            self.filenamesGt.extend(filenamesGt)
        self.filenames.sort()
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class NYUv2(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'images/')
        self.labels_root = os.path.join(root, 'labels40/')
        self.subset_ls = f'{root}{subset}.txt'
        print (self.images_root)
        with open(self.subset_ls) as f:
            dir_subset = [line.strip() for line in f.readlines()]
        self.filenames = [os.path.join(self.images_root,f'{dp}.jpg') for dp in dir_subset]
        self.filenamesGt = [os.path.join(self.labels_root,f'{dp}.png') for dp in dir_subset]
        self.filenames.sort()
        self.filenamesGt.sort()
        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        label=label.point(lambda p: p-1)
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)
    
class VOC2012(Dataset):

    def __init__(self, root, co_transform=None, subset='trainaug'):
        self.images_root = os.path.join(root, 'JPEGImages/')
        self.labels_root = os.path.join(root, 'SegmentationClassAugRaw/')
        self.subset_ls = os.path.join(root,f'ImageSets/Segmentation/{subset}.txt')
        print (self.images_root)
        with open(self.subset_ls) as f:
            dir_subset = [line.strip() for line in f.readlines()]
        self.filenames = [os.path.join(self.images_root,f'{dp}.jpg') for dp in dir_subset]
        self.filenamesGt = [os.path.join(self.labels_root,f'{dp}.png') for dp in dir_subset]
        self.filenames.sort()
        self.filenamesGt.sort()
        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        label = label.point(lambda p: 21 if p==255 else p)
        image = image.resize((224,224))
        label = label.resize((224,224))
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)
    
