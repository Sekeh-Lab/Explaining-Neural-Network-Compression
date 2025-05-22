import pandas as pd
import numpy as np
from pretrained import whole
from  AP2_computation import AP2_class
from Montecarlo import montecarlo
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import loadingdata
from AP3_Gaussian import AP3_Gaussian_class
import pickle
import os
from torch import nn
import copy

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class main():

    def __init__(self, pruned_epoch_number=20, trained_epoch_number=100, v=4, group=100, network='vgg16', data='CIFAR10', n_samples=600000, pretrained=True, pruned_layer="logit", AP3_method='gaussian'):
        self.pruned_epoch_number = pruned_epoch_number
        self.trained_epoch_number = trained_epoch_number
        self.dataframes = {}
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v= v
        self.group= group
        self.network= network
        self.data= data
        self.AP3_method = AP3_method
        self.n_samples= n_samples
        self.pretrained=pretrained
        self.base_path= f"./images_{self.network}_{self.data}_{self.AP3_method}_{self.trained_epoch_number}_{self.pruned_epoch_number}"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.dataset = loadingdata.Dataset(data)
        self.trainset, self.testset = self.dataset.data_reader()
        self.pruned_layer = pruned_layer
        

    def execute(self):
        Average={}       
        for itera in [1,2,3]:
            self.dataframes = {}
            
            # first train the pre-trained benchmark
            pretrain = whole(self.device, network_name=self.network, data=self.data, batch_size=128,
                                epoch_number=self.trained_epoch_number, pretrained=self.pretrained, itera=itera, trainset=self.trainset, testset=self.testset, pruned_layer=self.pruned_layer)
            
            if not os.path.exists(f"original_model_state_dict_trail{itera}_network{self.network}_data{self.data}.pt"):

                original_model, acc_data, vanila_concat_set =pretrain.function(trained_epoch_num=self.trained_epoch_number,
                                                                pruned_step='False',
                                                                percentage=None, method=None, model=pretrain.model,
                                                                mask=None)
                print('end of pretrein-function')
                try:
                    torch.save(original_model.state_dict(), f'original_model_state_dict_trail{itera}_network{self.network}_data{self.data}.pt')
                    torch.save(acc_data, f'acc_data_new_trail{itera}_network{self.network}_data{self.data}.pt')
                    torch.save(vanila_concat_set, f'vanila_concat_set_new_trail{itera}_network{self.network}_data{self.data}.pt')
                except:
                    pass

                
            else:

                state_dict = torch.load(f'./original_model_state_dict_trail{itera}_network{self.network}_data{self.data}.pt')
                original_model = pretrain.model
                original_model.load_state_dict(state_dict)
                # print('\n original_model is:', original_model)
                acc_data = torch.load(f'./acc_data_new_trail{itera}_network{self.network}_data{self.data}.pt')
                vanila_concat_set = torch.load(f'./vanila_concat_set_new_trail{itera}_network{self.network}_data{self.data}.pt')


                # state_dict = torch.load(f'./original_model_state_dict_trail{itera}_network{self.network}.pt')
                # original_model = pretrain.model
                # original_model.load_state_dict(state_dict)
                # # print('\n original_model is:', original_model)
                # acc_data = torch.load(f'./acc_data_new_trail{itera}_network{self.network}.pt')
                # vanila_concat_set = torch.load(f'./vanila_concat_set_new_trail{itera}_network{self.network}.pt')

            vanila_concat=vanila_concat_set['{}'.format(self.trained_epoch_number-1)]


            lambda_max=[]
            for pruned_method in ['lowest','highest','random']: 
        
                print('pruned_method is', pruned_method)

                for percentage in [0.1, 0.3, 0.5, 0.8]: 
                # for percentage in [0.3, 0.5,0.8]: 
                    print('percentage is :',percentage,'done\n\n\n')
                

                    variable_name = "{}-{}".format(pruned_method,percentage)

                    dataframe = pd.DataFrame(np.zeros((self.pruned_epoch_number, 4)), columns=['AP2', 'AP3', 'Performance_Difference_train', 'Performance_Difference_test'])
                    
                    model_clone = copy.deepcopy(original_model)

                    if not os.path.exists('pruned_model_trial{}_per_{}_{}_network{}_data{}.pt'.format(itera, percentage, pruned_method, self.network, self.data)):
                        #fine-tunning the pruned network
                        mask, pruned_model, pruned_acc_data, pruned_concat_set = pretrain.finetunning_pruned(
                            percentage=percentage, model=model_clone, pruned_epoch_number=self.pruned_epoch_number, pruned_method=pruned_method)

                        try:
                            torch.save(mask, f'./mask_file_trial{itera}_per_{percentage}_{pruned_method}_network{self.network}_data{self.data}.pt')
                            torch.save(pruned_model.state_dict(), 'pruned_model_trial{}_per_{}_{}_network{}_data{}.pt'.format(itera, percentage, pruned_method, self.network, self.data))
                            torch.save(pruned_acc_data, 'pruned_acc_data_trial{}_per_{}_{}_network{}_data{}.pt'.format(itera, percentage, pruned_method, self.network, self.data))
                            torch.save(pruned_concat_set, 'pruned_concat_set_trial{}_per_{}_{}_network{}_data{}.pt'.format(itera, percentage, pruned_method, self.network, self.data))
                        except:
                            pass
                        
                    else:
                        mask = torch.load(f'./mask_file_trial{itera}_per_{percentage}_{pruned_method}_network{self.network}_data{self.data}.pt')
                        # print('\n\n mask is',mask)
                        pruned_state_dict = torch.load( './pruned_model_trial{}_per_{}_{}_network{}_data{}.pt'.format(itera, percentage,pruned_method, self.network, self.data)) 
                        # Create a deep copy of the original model
                        pruned_model = copy.deepcopy(original_model)

                        # Applying pruned mask on the orignal model to have the pruned version 
                        for name, module in pruned_model.named_modules():
                            if isinstance(module, nn.Linear):
                                # print('\n name and module is', name , module)
                                #chech the fc or classifier layer based on the used model
                                if self.pruned_layer=='logit':
                                    layer_name = pretrain.name
                                elif self.pruned_layer=='last-dense-layer' and self.network == 'vit':
                                    layer_name = 'vit.encoder.layer.11.output.dense'

                                if layer_name in name: 
                                    weight = module.weight.data
                                    weight.mul_(mask)  # Apply the mask

                        # loading finetunned state dicts
                        pruned_model.load_state_dict(pruned_state_dict, strict=False)                         
                        pruned_acc_data = torch.load('pruned_acc_data_trial{}_per_{}_{}_network{}_data{}.pt'.format(itera, percentage, pruned_method, self.network, self.data))
                        pruned_concat_set = torch.load('pruned_concat_set_trial{}_per_{}_{}_network{}_data{}.pt'.format(itera, percentage, pruned_method, self.network, self.data))


                    # if you would like to have AP3 based on Gaussian with non-diagonal covaraince, uncomment line 62 and comment line 63.
                    # Note that for alexnet and vgg as the covariance is too large, you need more RAM, CPU and GPU to construct it.
                    if self.AP3_method =='gaussian':
                        AP3_list = AP3_Gaussian_class(self.device, vanila_concat, pruned_concat_set, self.pruned_epoch_number).AP3_function()
                    else:
                        AP3_list=montecarlo(self.device, vanila_concat, pruned_concat_set, itera, self.n_samples, self.v, self.group, self.pruned_epoch_number).AP3_computation()
                    # print(f'method{pruned_method}-per{percentage}_AP3 value is',AP3_list)
                    AP2_list=[AP2_class(self.device, vanila_concat,pruned_concat_set['{}'.format(i)]).AP2_function() for i in range (self.pruned_epoch_number)]
                    PD_train_list=[np.abs(acc_data.iloc[0,self.trained_epoch_number-1]-pruned_acc_data.iloc[0,i]) for i in range(self.pruned_epoch_number)]
                    PD_test_list=[np.abs(acc_data.iloc[1,self.trained_epoch_number-1]-pruned_acc_data.iloc[1,i]) for i in range(self.pruned_epoch_number)]
                    dataframe['AP3']= AP3_list
                    dataframe['AP2']= AP2_list
                    dataframe['Performance_Difference_train']= PD_train_list
                    dataframe['Performance_Difference_test']= PD_test_list
                    
                    # in order to finc the suitable coffeicient of identity matrix as the covariance matrix for Gaussian, we need the following variable.
                    lambda_max.append(AP2_list)
                    self.dataframes[variable_name] = dataframe
                    # model = model

            self.dataframes = self.modify_first_column(lambda_param=lambda_max, DataFrame=self.dataframes)
            try:
                df = pd.DataFrame(self.dataframes)
                df.to_csv(f"./dataframe_trial{itera}_network{self.network}.csv", index=False)
            except Exception as e:
                print(f"Error converting dictionary to DataFrame or saving CSV: {e}")
                
            Average['{}'.format(itera)] = self.dataframes
        
        final_datasets={'{}-{}'.format(i, j): (sum(Average['{}'.format(trial)]['{}-{}'.format(i, j)] for trial in range(1, itera + 1)) / len(range(1, itera + 1)))
                         for i in ['lowest','highest','random'] for j in [0.1,0.3,0.5,0.8]}
        


        with open(f'final_datasets_network{self.network}_{self.data}_structured.pkl', 'wb') as pkl_file:
            pickle.dump(final_datasets, pkl_file)

        
        return final_datasets
        

    def modify_first_column(self, lambda_param=None,DataFrame=None):
        '''
        this function helps to find the covariance matrix of gaussian in a way that all AP2 values are less that 2.
        '''
        for key in DataFrame.keys():
            DataFrame[key]['AP2'] = DataFrame[key]['AP2'] * ((2*1.9)/(np.max(lambda_param)))

        return DataFrame

    def plot_test(self):
        
        '''
        ploting figures
        '''

        DataSet=self.execute()
    

        # param can be 'AP2' or 'AP3'
        
        for param in ['AP2','AP3']:
            for method in ['lowest','highest','random']: #

                fig, ax = plt.subplots()
                colors = ['red', 'blue', 'orange', 'purple']
                for i, percentage in enumerate([0.1,0.3,0.5,0.8]):
                    data1=DataSet['{}-{}'.format(method, percentage)]
                    data1.index = range(1, self.pruned_epoch_number+1)
                    if param=='AP2':
                        #AP2
                        ax.plot(data1.iloc[:self.pruned_epoch_number+1, 0], linewidth=3, color=colors[i])
                    else:
                        #AP3
                        ax.plot(data1.iloc[:self.pruned_epoch_number+1, 1], linewidth=3, color=colors[i])
                    #Performance difference on test data
                    ax.plot(data1.iloc[:self.pruned_epoch_number+1, 3], linewidth=3, linestyle='--', color=colors[i])

                # Axis formatting
                # ax.xaxis.grid(True)
                # ax.yaxis.grid(True)
                ax.set_yscale("log")
                ax.set_xlabel("Epoch")
                ax.set_xticks(range(1, self.pruned_epoch_number + 1))
                ax.set_xticklabels(range(1, self.pruned_epoch_number+1))
                save_path = Path(
                    '{}-{}-log-all percentages-method{}-param{}-kl-{}-cifar-test.png'.format(self.v, self.group, method,param,
                                                                                            self.network))
                fig.savefig(self.base_path/save_path, format='png')


            colors = ['red', 'orange', 'green']
            for percentage in [0.1, 0.3, 0.5, 0.8]:
                fig, ax = plt.subplots()
                for i, method in enumerate(['lowest','highest','random']):
                    data1 = DataSet['{}-{}'.format(method, percentage)]
                    data1.index = range(1, self.pruned_epoch_number+1)
                    if param=='AP2':
                        #AP2
                        ax.plot(data1.iloc[:self.pruned_epoch_number+1, 0], linewidth=3, color=colors[i])
                    else:
                        #AP3
                        ax.plot(data1.iloc[:self.pruned_epoch_number+1, 1], linewidth=3, color=colors[i])
                    # Performance difference on test data
                    ax.plot(data1.iloc[:self.pruned_epoch_number+1, 3], linewidth=3, linestyle='--', color=colors[i])

                # ax.xaxis.grid(True)
                # ax.yaxis.grid(True)
                ax.set_yscale("log")
                ax.set_xlabel("Epoch")
                ax.set_xticks(range(1, self.pruned_epoch_number + 1))
                ax.set_xticklabels(range(1, self.pruned_epoch_number+1))
                save_path = Path(
                    '{}-{}-log-all methods-param{}-percentage{}-kl-{}-cifar-test.png'.format(self.v, self.group, param,percentage, self.network))
                fig.savefig(self.base_path/save_path, format='png')
        return
    




if __name__ == '__main__':
    '''
    data = ['CIFAR10', 'CIFAR100', 'tiny_imagenet']
    net = ['resnet', 'alexnet', 'vgg16', 'vit']
    AP3_method = 'gaussian' or 'T-student'
    v is the hyperparameter for T-student that show the skew of the distibition. For large value of v, T-student gets closer to a gaussian distribution
    pruned_layer = In all expriments we have "logits", for vit the other layer named "last-dense-layer" is also examined. 
    You could change lines 124-127 and add other layers to be pruned.
    
    '''
    for net in ['resnet']:
        for data in ['tiny_imagenet']:
            print('data is:',data,'\n\n')
            s = main( pruned_epoch_number=20, trained_epoch_number=60, v=4, group=100, network=net, data=data, n_samples=600000, pretrained=True, pruned_layer="logit", AP3_method='T-student')
            s.plot_test()



