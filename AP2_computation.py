import torch

class AP2_class():

    '''
    Computing AP2 of two vectors

    '''
    def __init__(self, device, concat1=None, concat2=None):
        self.device = device
        self.concat1 = concat1
        self.concat2 = concat2
        self.concat1 = self.concat1.to(self.device)
        self.concat2 = self.concat2.to(self.device)

    def AP2_function(self):
        return torch.norm((self.concat1-self.concat2), p=2).item()**2
