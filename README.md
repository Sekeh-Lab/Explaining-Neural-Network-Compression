# Explaining-Neural-Network-Compression

README – Supplementary Material for [Towards Explaining Deep Neural Network Compression Through a Probabilistic Latent Space]

This supplementary material includes the Python code used to generate the experimental results in the paper. The code implements various pruning methods and computes the AP2 and AP3 metrics described in the manuscript.
If you are intrerested and willing to use the code, Please find the published version of the paper at ACM TOPML Journal to cite.

--------------------------------------------------------------------------------
CONTENTS
--------------------------------------------------------------------------------

1. main.py  
   - The main entry point for running experiments.  
   - Supports selecting different models, pruning methods, and pruning percentages.

2. pretrained.py  
   - Handles training of the original (unpruned) network.  
   - Also includes training routines for pruned networks.

3. pruning_methods.py  
   - Contains implementations of various pruning strategies used in the study.

4. modelfile.py  
   - Defines and loads pretrained model architectures.

5. loadingdata.py  
   - Loads all datasets used in the experiments.

6. generating_sample.py  
   - Used for generating samples required in the Monte Carlo estimation method.

7. Montecarlo.py  
   - Computes AP3 values using the multivariate Student-t distribution.

8. AP3_gaussian.py  
   - Computes AP3 values assuming a multivariate Gaussian distribution.

9. AP2_computation.py  
   - Computes AP2 metric.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

To reproduce main results:
    python main.py 
Refer to `main.py` for all available arguments.

----------------------------------------------------------------------------

