# AccSGD
Implements pytorch code for the Accelerated SGD algorithm.

This is the code associated with the paper ``On the insufficiency of existing momentum schemes for Stochastic Optimization'', appearing at ICLR 2018. 

Usage:

The code can be downloaded and placed in a given local directory. In a manner similar to using any usual optimizer from the pytorch toolkit, it is also possible to use the AccSGD optimizer with little effort.
First, we require importing the optimizer through the following command:

from AccSGD import *

Next, an ASGD optimizer working with a given pytorch model can be invoked using the following command:

optimizer = AccSGD(model.parameters(), lr=0.1, kappa = 1000.0, xi = 10.0)

where, lr is the learning rate, kappa the long step parameter and xi is the statistical advantage parameter.

Rough guidelines: 

As the batch size increases by a factor of "k", increase xi by sqrt(k).

As the networks grow deeper (specifically with Resnets and beyond), employing kappa to be 10^4 or more helps. For shallow nets, a typical value of kappa resembles 10^3 or even 10^2.

Citation:

If AccSGD is used in your paper/experiments, please cite the following papers.


@inproceedings{Kidambi2018Insufficiency,
  title={On the insufficiency of existing momentum schemes for Stochastic Optimization},
  author={Kidambi, Rahul and Netrapalli, Praneeth and Jain, Prateek and Kakade, Sham},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
@Article{Jain2017Accelerating,
  title =	"Accelerating Stochastic Gradient Descent",
  author =	"Prateek Jain and Sham M. Kakade and Rahul Kidambi
		 and Praneeth Netrapalli and Aaron Sidford",
  journal =	"CoRR",
  year = 	"2017",
  volume =	"abs/1704.08227"
}