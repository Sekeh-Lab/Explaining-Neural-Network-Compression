# Data

from torchvision import datasets, transforms
import os
import numpy as np
import requests
import zipfile
import io
import torch


class Dataset():
    def __init__(self, data='CIFAR10'):
        self.data = data

    def data_reader(self):
        trainset = None  # Initialize trainset outside the conditional blocks
        testset = None   # Initialize testset outside the conditional blocks
        # self.data = 'CIFAR10'   #changed
        if self.data == 'CIFAR10':
            print('==> Preparing data..')
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),  #first crop the image randomly and then resize it.
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
                ])

            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            trainset = datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)

            testset = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)

        # else:  #changed
        elif self.data == 'CIFAR100':
            print('==> Preparing data..')
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),  #first crop the image randomly and then resize it.
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
            trainset = datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)

            testset = datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)
            
        elif self.data== 'tiny_imagenet':
            print('==> Preparing data..')

            '''
            #### Uncomment these lines if the val data have no target
            import os
            import re
            def create_dir(base_path, classname):
                path = base_path + classname
                if not os.path.exists(path):
                    os.mkdir(path)

            def reorg(filename, base_path, wordmap):
                #print(len(wordmap))
                with open('./data/tiny-imagenet-200/val/val_annotations.txt') as vals:
                    for line in vals:
                        vals = line.split()
                        imagename = vals[0]
                        #print(vals[1])
                        classname = wordmap[vals[1]]
                        if os.path.exists(base_path+imagename):
                            #print(base_path+imagename, base_path+classname+'/'+imagename)
                            os.rename(base_path+imagename,  base_path+classname+'/'+imagename)


            wordmap = {}
            with open('./data/tiny-imagenet-200/words.txt') as words, open('./data/tiny-imagenet-200/wnids.txt') as wnids:
                for line in wnids:
                    vals = line.split()
                    wordmap[vals[0]] = ""
                for line in words:
                    vals = line.split()
                    if vals[0] in wordmap:
                        single_words = vals[1:]
                        classname =  re.sub(",", "", single_words[0])
                        if len(single_words) >= 2:
                            classname += '_'+re.sub(",", "", single_words[1])
                        wordmap[vals[0]] = classname
                        create_dir('./data/tiny-imagenet-200/val/images/', classname)
                        if os.path.exists('./data/tiny-imagenet-200/train/'+vals[0]):
                            os.rename('./data/tiny-imagenet-200/train/'+vals[0], './data/tiny-imagenet-200/train/'+classname)
                        #create_dir('./test/images/', single_words[0])
                        #create_dir('./train/images/', single_words[0])


            reorg('./data/tiny-imagenet-200/val/val_annotations.txt', './data/tiny-imagenet-200/val/images/', wordmap)
            '''


            directory = "./data/tiny-imagenet-200"
            num_classes = 200

            transform_mean = np.array([ 0.485, 0.456, 0.406 ])
            transform_std = np.array([ 0.229, 0.224, 0.225 ])
             
            #transform_mean = np.array([ 0.4802, 0.4481, 0.3975]) # I got
            #transform_std = np.array([ 0.2296, 0.2263, 0.2255]) # I got

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                transforms.Normalize(mean = transform_mean, std = transform_std),
            ])

            test_transform = transforms.Compose([
               transforms.Resize(224),
                    transforms.ToTensor(),
                transforms.Normalize(mean = transform_mean, std = transform_std),
            ])

            traindir = os.path.join(directory, "train")
            # be careful with this set, the labels are not defined using the directory structure
            testdir = os.path.join(directory, "val/images")
    
            trainset = datasets.ImageFolder(traindir, train_transform)
            #print(trainset)
            testset = datasets.ImageFolder(testdir, test_transform)
            
        return trainset, testset


