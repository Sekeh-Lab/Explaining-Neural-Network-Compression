import torch
from torch import nn
import numpy as np
from torch.nn.utils import prune


class pruning_method():

    '''
        Three different magnitude based pruning methods are defined here. 
    '''
    def __init__(self, method='lowest',modell=None, model_name='vgg16', percentage=0.1, device= None, pruned_layer="logit"):
        self.method = method
        self. modell = modell
        self.percentage = percentage
        self.device = device
        self.model_name = model_name
        self.pruned_layer = pruned_layer

        if self.model_name in ['vgg16','alexnet']:
            self.model_layer = self.modell.classifier[6]
            self.layer_weight = 'classifier.6.weight_mask'
            self.name= 'classifier.6'
        elif self.model_name == 'resnet':
            self.model_layer = self.modell.fc
            self.layer_weight = 'fc.weight_mask'
            self.name = 'fc'
        elif self.model_name == 'vit':
            if self.pruned_layer=='logit':
                print('\n layer which is pruning: logit')
                self.model_layer = self.modell.classifier
                self.layer_weight = 'classifier.weight_mask'
                self.name = 'classifier'

            elif self.pruned_layer=='last-dense-layer':
                print('\n layer which is pruning: last-dense-layer')
                # removing dense layer with shape (3073,768)
                self.model_layer = self.modell.vit.encoder.layer[-1].output.dense  # Dense layer in the last ViTLayer
                self.layer_weight = 'vit.encoder.layer.11.output.dense.weight_mask'  # Adjusting for weight mask
                self.name = 'encoder.layer.11.output.dense'
                
            # elif self.args.pruned_layer=='last-attenstion-key-layer':

    def run(self):
        if self.method == 'lowest':
            return self.lowest()
        elif self.method == 'highest':
            return self.highest()
        elif self.method == 'random':
            return self.random()
        else:
            return self.structured()

    def lowest(self):

        # pruning algorithm for lowest magnitude based
        pruned = prune.l1_unstructured(self.model_layer, name='weight', amount=self.percentage)
        mask = self.modell.state_dict()[self.layer_weight]  # mask matrix to prune the weights of the last layer based on
        return mask, self.modell

    def highest(self):

        abs_linear = torch.abs(self.model_layer.weight)
        max_magnitude = torch.max(abs_linear)
        max_tensor = torch.full_like(abs_linear, max_magnitude.item())
        self.model_layer.weight.data = abs_linear - max_tensor
        self.modell = self.modell.to(self.device)
        pruned = prune.l1_unstructured(self.model_layer, name='weight', amount=self.percentage)
        return self.modell.state_dict()[self.layer_weight], self.modell

    def random(self):

        mask = torch.ones_like(self.model_layer.weight.data)
        num_elements = np.prod(mask.shape)
        num_zeros = int(np.round(self.percentage * num_elements))

        # Set the random seed for reproducibility (optional)
        np.random.seed(42)

        # Generate random indices without replacement
        random_indices = np.random.choice(num_elements, size=num_zeros, replace=False)

        # Reshape the indices to match the tensor shape
        row_indices, column_indices = np.unravel_index(random_indices, mask.shape)

        # Set the selected elements to zero
        mask[row_indices, column_indices] = 0
        self.model_layer.weight.data = self.model_layer.weight.data * mask
        self.modell = self.modell.to(self.device)
        return mask, self.modell



    def structured(self):
        print('\n structured_pruning')
        # Structured pruning by removing entire channels/neurons
        # if isinstance(self.model_layer, nn.Conv2d):
            # prune.ln_structured(self.model_layer, name='weight', amount=self.percentage, n=2, dim=0)  # Prune channels
        # elif isinstance(self.model_layer, nn.Linear):
        prune.ln_structured(self.model_layer, name='weight', amount=self.percentage, n=1, dim=0)  # Prune neurons

        mask = self.modell.state_dict()[self.layer_weight]
        return mask, self.modell
    


    def check(self, verbose=False,model=None):

        """
            This function helps to see how many of weights in the last layer is pruned using the pruning method
        """
        print('Checking...')
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%s: Pruned %d/%d (%.2f%%)' %
                          (name, num_zero, num_params, 100 * num_zero / num_params))





