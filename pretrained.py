import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import modelfile   # python file to call pre-trained benchmarks
import pruning_methods  # python file to call magnitude-based pruning methods
from itertools import islice
import copy
# from DataGenerator import tiny_imagenet_DataGenerator
from torchvision import transforms
import numpy as np


class whole():

    def __init__(self, device, network_name='vgg16', data='CIFAR10', batch_size=128, epoch_number=100, pretrained=True, itera=1, trainset=None,testset=None, pruned_layer='logit'):
        self.device = device
        self.trainset=trainset
        self.testset=testset
        self.epoch_number = epoch_number
        self.network_name = network_name
        self.data = data
        self.flag = pretrained
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.itera=itera
        self.concatset = {}
        self.weight_decay = 0
        self.pruned_layer = pruned_layer
        # self.train_test_accuracy_epochs = pd.DataFrame([[0 for i in range(self.epoch_number)],
        #                                                 [0 for i in range(self.epoch_number)]],
        #                                                columns=[i for i in range(self.epoch_number)])

        self.batch_size = batch_size

        self.network = modelfile.Network(self.device, self.network_name, self.flag)
        self.model = self.network.set_model()
        # print('model is', self.model)
        if self.network_name in ['vgg16', 'alexnet']:
            self.lr = 0.01
            self.name = 'classifier.6'
            if self.data == 'CIFAR10':
                self.model.classifier[6] = nn.Linear(4096, 10)
            elif self.data == 'CIFAR100': #mahsa
                self.model.classifier[6] = nn.Linear(4096, 100)
            else: #mahsa
                self.lr = 0.001
                self.weight_decay = 0.0001
                self.model.classifier[6] = nn.Linear(4096, 200)
        elif self.network_name == 'resnet':
            self.lr = 0.1
            self.name = 'fc'
            if self.data == 'CIFAR10':
                self.model.fc = nn.Linear(2048, 10)
            elif self.data == 'CIFAR100': #mahsa
                self.model.fc = nn.Linear(2048, 100)
            else:
                self.model.fc = nn.Linear(2048, 200)
                self.lr = 0.001
                self.weight_decay = 0.0001
        elif self.network_name == 'vit':
            self.lr = 0.002  # Adjust learning rate for ViT
            self.name = 'classifier'
            num_classes = 10 if self.data == 'CIFAR10' else 100 if self.data == 'CIFAR100' else 200
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)  # Adjust the final layer for your dataset

        self.model=self.model.to(self.device)

        torch.manual_seed(self.itera)
     
        self.xx = torch.utils.data.DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=2)
        self.yy = torch.utils.data.DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=2)
        

    def function(self, trained_epoch_num= None, pruned_step='False', percentage= None, method = None, model= None, mask=None):
        print('enter function')
        train_test_accuracy_epochs = pd.DataFrame([[0 for i in range(trained_epoch_num)],
                                                        [0 for i in range(trained_epoch_num)]],
                                                       columns=[i for i in range(trained_epoch_num)])

        '''
        here, the aim is calling train and test function in each opech 
        in order to train the pre trained benchmarks or fine tunne the pruned networks based on 
        the parameter pruned_step
        
        '''
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay = self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=trained_epoch_num)
        self.concatset={}

        print('trained_epoch_num is', trained_epoch_num, '\n\n')
        for epoch in range(trained_epoch_num):
            print('pretrain epoch is', epoch,'\n\n')
            concat_name = '{}'.format(epoch)

            # Training the pre-trained model (if pruned_step is False) or fine-tunning the pruning one (pruned_step is True)
            model, dataframe = self.train(model=model, epoch=epoch, xx=self.xx, pruned_step=pruned_step, mask=mask, dataframe=train_test_accuracy_epochs)
            # Training the model
            dataframe = self.test(model, epoch, self.yy, dataframe=dataframe)
            self.scheduler.step()


            # As all expriments are on th last layer of the network, I used the last layer to be vectorized.
            if self.network_name in ['vgg16','alexnet']:
                concat = model.classifier[6].weight.flatten()
                self.concatset[concat_name] = concat
            
            elif self.network_name == 'resnet':
                concat = model.fc.weight.flatten()
                self.concatset[concat_name] = concat

            elif self.network_name == 'vit':
                concat = model.classifier.weight.flatten()
                self.concatset[concat_name] = concat
                
        return model , dataframe, self.concatset

    # training function
    def train(self, model, epoch, xx, pruned_step='False', mask=None, dataframe=None):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(xx):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = model(inputs)
            if self.network_name == 'vit':
                outputs = outputs.logits  # Extract the logits from the output objec
            loss = self.criterion(outputs, targets)
            loss.backward()

            # fine-tunning the pruned model in order to just update the non-pruned weights in the last layer
            if pruned_step == 'True':
                for name, module in islice(model.named_modules(), 1, None):
                    try:
                        if name != self.name:
                            module.weight.grad = torch.zeros_like(module.weight.grad)
                        else:
                            module.weight.grad[mask == 0] = 0
                    except:
                        continue

            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        dataframe.loc[:, epoch] = None
        dataframe.iloc[0, epoch] = 100. * correct / total

        print(epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / len(xx),  100. * correct / total,
                                                           correct, total))
        return model, dataframe

    # testing function
    def test(self, model, epoch, yy, dataframe=None):

        # testing
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(yy):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                if self.network_name == 'vit':
                    outputs = outputs.logits  # Extract the logits from the output objec
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        dataframe.iloc[1, epoch] = 100. * test_correct / test_total
        print(epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss / len(yy), 100. * test_correct / test_total, test_correct, test_total))

        return dataframe

    def finetunning_pruned(self, percentage, model,  pruned_epoch_number, pruned_method):

        '''
        :param percentage: percentage to be pruned
        :param pruned_epoch_number: number of epochs so the pruned model will be finetuned based on
        :param pruned_method: method of pruning (lowest, highest, random)
        :return: accuracies and performance difference of every epoch
        
        step 1= then, the returned model wii be sent to pruning_method file to be pruned based on given method and percentage.

        step 2= the pruned model will be passed to function to be fined tuned. This step will be reapeted for pruned_epoch_num (20)
        epochs. In each iteration, the AP2, AP3 and performance difference of that epoch (considering concat of trained vanilla)
        will be computed.
        '''
        print('\n model is', model)
        copied_model = copy.deepcopy(model)
        copied_model = copied_model.to(self.device)
        # step 1
        func = pruning_methods.pruning_method(method=pruned_method, modell=copied_model, model_name=self.network_name,
                                              percentage=percentage, device=self.device, pruned_layer=self.pruned_layer)
        mask, copied_model = func.run()
        
        # step 2
        func.check(model=copied_model, verbose='True')
        copied_model, pruned_acc_data, pruned_concat_set = self.function(trained_epoch_num=pruned_epoch_number, pruned_step='True',
                                                       percentage=percentage, method=pruned_method, model=copied_model, mask=mask)

        return mask, copied_model, pruned_acc_data, pruned_concat_set






