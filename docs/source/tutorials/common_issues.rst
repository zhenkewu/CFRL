Common Issues
=======================

This section introduces some common issues in CFRL and discusses how 
users might deal with these issues.

Non-convergence
-----------------------

Non-convergence happens when the model fails to converge 
at the end of model training. In CFRL, non-convergence can arise when

1. Training the :code:`SequentialPreprocessor`.
2. Training the :code:`SimulatedEnvironment`.
3. Training the :code:`FQI`.
4. Training the :code:`FQE` in :code:`evaluate_reward_through_fqe()`.

Sources of Non-convergence
~~~~~~~~~~~~~~~~~~~~~~~

Non-convergence can happen in :code:`SequentialPreprocessor` and 
:code:`SimulatedEnvironment` training because they rely on training a 
neural network or a polynomial regression model to approximate the 
underlying environment's transition dynamics. If the neural network or 
the polynomial regression model fail to converge, then non-convergence 
arises. 

Similarly, non-convergence can happen in :code:`FQI` and :code:`FQE` 
training because they rely on training a neural network or a polynomial 
regression model to approximate the Q function, and non-convergence in 
the Q function approximator model can give rise to non-convergence in  
:code:`FQI` and :code:`FQE`. Moreover, there is one more source of 
non-convergence in :code:`FQI` and :code:`FQE`. The Q function approximator 
is updated in each iteration of FQI and FQE training. Thus, non-convergence 
can also arise if the approximated Q function fails to converge to the true 
Q function in these iterative updates.

Checking for Non-convergence Using Loss Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

CFRL provides a simple loss monitoring tool that can check for potential non-convergence 
in the training of neural networks in :code:`SequentialPreprocessor`, 
:code:`SimulatedEnvironment`, :code:`FQI`, and :code:`FQE`. This tool can 
be enabled by setting :code:`is_loss_monitored=True` in 
:code:`evaluate_reward_through_fqe()` and in the constructors of 
:code:`SequentialPreprocessor`, :code:`SimulatedEnvironment`, :code:`FQI`, 
and :code:`FQE`. When this tool is enabled, it divides the training data into 
a training set and a validation set and then monitors the validation loss in 
each epoch of neural network training. It raises a warning if the decrease 
in the validation loss is greater than some threshold in at least one of the 
final :math:`p` epochs of training, where the threshold and :math:`p` are 
user-specified parameters.

We note that while loss monitoring might flag potential non-convergence, it is 
neither necessary nor sufficient for non-convergence. For example, when the 
validation loss plateaus without reaching the true minimum, loss monitoring will 
not raise warnings despite non-convergence. On the other hand, if the choice of 
the threshold is too small, or the choice of :math:`p` is too large, then the 
tool will raise a warning even though the model has almost converged. Moreover, 
this loss monitoring tool is applicable only when the model type is :code:`"nn"`, 
and it only checks for potential non-convergence in neural network training 
rather than potential non-convergence in the update of the Q function approximators.

Reducing Non-convergence
~~~~~~~~~~~~~~~~~~~~~~~~

To reduce the likelihood of non-convergence in neural network training, users can 
try increasing the maximum number of training epochs, adjusting the learning rate, 
or increasing the size of the training data. To reduce the likelihood of 
non-convergence in the update of the Q function approximators, users can try 
increasing the maximum number of FQI/FQE iterations or ensuring the training data 
covers a sufficiently large portion of the state and action spaces.