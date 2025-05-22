import torch
import warnings
warnings.filterwarnings('ignore')
import pyro.distributions as dist


class Samples():

    '''
    generating samples from multivaraite T-student distribution
    '''
    def __init__(self, n_samples, v):
        self.n_samples = n_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v = v

    def samples(self, mean):
        d = mean.shape[0]
        sigma = torch.eye(d).to(self.device)
        distribution = dist.MultivariateStudentT(loc=mean, scale_tril=sigma, df=self. v)
        sample_dist = distribution.sample(sample_shape=torch.Size([self.n_samples]))
        return (distribution, sample_dist)
