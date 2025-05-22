import torch
import warnings
warnings.filterwarnings('ignore')
from generating_sample import Samples


class montecarlo():
    def __init__(self,device, concat1, concat2, itera, n_samples, v, group, epochnum=1):
        self.device = device
        self.concat1 = concat1
        self.concat2=concat2
        self.itera=itera
        self.n_samples=n_samples
        self.v=v
        self.group=group
        self.epochnum=epochnum

    def AP3_computation(self):
        torch.manual_seed(self.itera)
        self.concat1=self.concat1.to(self.device)
        # norm = torch.norm(self.concat1, p=2)
        # self.concat1 = self.concat1/ (norm + 1e-6) 
        group_size = self.concat1.size(0) // self.group
        

        # Create a list to store the grouped tensors
        grouped_tensors = []

        # Group the tensor
        for i in range(0, self.concat1.size(0), group_size):
            group_tensor = self.concat1[i: i + group_size]
            grouped_tensors.append(group_tensor)

        group_means = [group.mean().item() for group in grouped_tensors]
        group_means = torch.tensor(group_means).to(self.device)
        # **Log the group means for concat1**
        
        point = Samples(self.n_samples, self.v)

        distp, samplep = point.samples(group_means)

        KL = []
        for epoch in range(self.epochnum):
          
            weight_pruned = self.concat2['{}'.format(epoch)]
            weight_pruned = weight_pruned .to(self.device)
            # norm = torch.norm(weight_pruned, p=2)
            # weight_pruned = weight_pruned/ (norm + 1e-6)
            group_size = weight_pruned.size(0) // self.group



            # Create a list to store the grouped tensors
            grouped_tensors = []

            # Group the tensor
            for i in range(0, weight_pruned.size(0), group_size):
                group_tensor = weight_pruned[i: i + group_size]
                grouped_tensors.append(group_tensor)


            group_means_method = [group.mean().item() for group in grouped_tensors]
            group_means_method = torch.tensor(group_means_method).to(self.device)

    
            cosine_sim = torch.nn.functional.cosine_similarity(group_means.to(self.device), group_means_method.to(self.device), dim=0)
            diff = torch.abs(group_means - group_means_method)
            
            mean = 0
            for j in range(100):  #number of iterations
                distq,sampleq=point.samples(group_means_method)
                dp = distp.log_prob(sampleq)
                dq = distq.log_prob(sampleq)
                kl = dq-dp
                # print("kssssssl:", kl)
                # kl = kl.sum(-1)
                kl = kl.mean(-1)
                # print("kl:", kl)
                # mean = mean+kl.sum()
                mean = mean+kl.mean().item()
            # mean = mean/self.n_samples
            KL.append(mean/100)
            # print('\n KL-divergence is', KL, flush=True)
        return KL
            



