import torch
import numpy as np

class AP3_Gaussian_class():

    '''
    Computing AP3 using Gaussian with non-diagonal covariance matrices

    '''
    def __init__(self, device, concat1=None, concat2=None, epochnum=1):
        self.device=device
        self.concat1 = concat1
        self.concat1 = self.concat1.to(self.device)
        self.concat2 = concat2
        self.epochnum= epochnum

    def positive_definite_covariance(self, dim):
        '''
        This function generates a non-diagonal covariance matrix. Note that all coef are arbitrary to have small AP3s.
        '''
        np.random.seed(27)
        while True:
            random_matrix = torch.rand(dim, dim)
            # Construct the covariance matrix with non-diagonal elements
            cov_matrix = (random_matrix + random_matrix.t()) * 0.01
            # Compute eigenvalue decomposition
            eig_vals, eig_vecs = torch.linalg.eigh(cov_matrix)

            # eig_vals[:]=a+1
            eig_vals[:] = 0.01  # this is optional. I did that to have small AP3 values

            cov_matrix_inverse = (1 / (np.sqrt(dim))) * 0.1 * (eig_vecs @ torch.diag(eig_vals) @ eig_vecs.t())  # The coeeficients here are optional.
            print(cov_matrix_inverse)
            rows, cols = cov_matrix_inverse.shape
            # the following lines check that if the inverse matrix is non diagonal
            for i in range(rows):
                for j in range(cols):
                    if i != j and cov_matrix_inverse[i, j] != 0:
                        non_diagonal_found = True
                        break
                if non_diagonal_found:
                    break
            if non_diagonal_found:
                break
        return cov_matrix_inverse

        # you can use any other non-diagonal covariance matrix here that needs less computation.
    def mahalanobis_distance(self, pruned_weight, covariance_matrix):

        # Calculate the Mahalanobis distance between x and y using the given covariance matrix
        diff = self.concat1-pruned_weight
        diff = diff.to('cpu').detach().numpy()
        covariance_matrix_cpu = covariance_matrix.to('cpu').detach().numpy()
        # inverse_covariance = np.linalg.inv(covariance_matrix.to('cpu'))
        mahalanobis_dist = np.sqrt(np.dot(np.dot(diff.T, covariance_matrix_cpu), diff))
        return mahalanobis_dist

    def AP3_function(self):
        covariance_inverse = self.positive_definite_covariance(self.concat1.size(0))
        AP3 = []
        for epoch in range(self.epochnum):
            pruned_weight = self.concat2['{}'.format(epoch)]
            pruned_weight = pruned_weight.to(self.device)
            kl_div = self.mahalanobis_distance(pruned_weight, covariance_inverse)
            AP3.append(kl_div)
        return AP3

