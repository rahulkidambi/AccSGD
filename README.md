# AccSGD
This is the code associated with Accelerated SGD algorithm used in the paper [On the insufficiency of existing momentum schemes for Stochastic Optimization](https://openreview.net/forum?id=rJTutzbA-), that is selected to appear at ICLR 2018.
## Usage:
The code can be downloaded and placed in a given local directory. In a manner similar to using any usual optimizer from the pytorch toolkit, it is also possible to use the AccSGD optimizer with little effort.
First, we require importing the optimizer through the following command:
```
from AccSGD import *
```
Next, an ASGD optimizer working with a given pytorch model can be invoked using the following command:
```
optimizer = AccSGD(model.parameters(), lr=0.1, kappa = 1000.0, xi = 10.0)
```
where, `lr` is the learning rate, `kappa` the long step parameter and `xi` is the statistical advantage parameter.
## Guidelines on setting parameters/debugging: 
*The learning rate* `lr`: `lr` set similar to schemes such as vanilla Stochastic Gradient Descent (SGD)/Standard Momentum (Heavy Ball)/Nesterov's Acceleration.

*Long Step* `kappa`: As the networks grow deeper (e.g. with resnets) and when dealing with typically harder datasets such as CIFAR/ImageNet, employing `kappa` to be 10^4 or more helps. For shallow nets and easier datasets such as MNIST, a typical value of `kappa` can be set as 10^3 or even 10^2.

*Statistical Advantage Parameter* `xi`: `xi` lies between 1.0 and `sqrt(kappa)`. When large batch sizes (nearly matching batch gradient descent) are used, it is advisable to use `xi` that is closer to `sqrt(kappa)`. In general, as the batch size increases by a factor of `k`, increase `xi` by `sqrt(k)`. 

Effective ways to debug:

For Nets with ReLU/ELU type activations:

(--1--) Slower convergence: There are three reasons for this to happen:
* This could be a result of setting the learning rate too low (similar to SGD/vanilla momentum/Nesterov's acceleration). 
* This could be as a result of setting `kappa` to be too high. 
* The other reason could be that `xi` can be set to a very small value.

(--2--) Oscillatory behavior/Divergence: There are two reasons for this to happen:
* This could be a result of setting the learning rate to be too high (similar to SGD/vanilla momentum/Nesterov's acceleration).
* The other reason is that `xi` can be set to a very large value and may require tuning down.

For nets with Sigmoid activations:

Slower convergence after an initial rapid decrease in error: This is a sign of an over aggressive setting of parameters and must be treated in a similar manner as the oscillatory/divergence behavior (--2--) encountered in the ReLU/ELU activation case.

Slow convergence right from the start: This is more likely related to slower convergence (--1--) encountered in the ReLU/ELU case.

## Citation:
If AccSGD is used in your paper/experiments, please cite the following papers.
```
@inproceedings{Kidambi2018Insufficiency,
  title={On the insufficiency of existing momentum schemes for Stochastic Optimization},
  author={Kidambi, Rahul and Netrapalli, Praneeth and Jain, Prateek and Kakade, Sham},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@Article{Jain2017Accelerating,
  title={Accelerating Stochastic Gradient Descent},
  author={Jain, Prateek and Kakade, Sham and Kidambi, Rahul and Netrapalli, Praneeth and Sidford, Aaron},
  journal={CoRR},
  year={2017},
  volume = {abs/1704.08227}
}
```
