.. _data_requirements:

Data Requirements
=============================

For CFRL to work properly, the offline trajectory data should satisfy some requirements, which are 
summarized below: 

1. **Sensitive Attribute:** The sensitive attribute can be either univariate or multivariate. Each 
component of the sensitive attribute vector can also be either numerical or categorical. If a component 
of the sensitive attribute is categorical, it should either be reparametrized as discrete numerical 
levels (e.g. 0="Race A", 1="Race B") or one-hot encoded manually.

2. **State Variable:** The state variable can either be univariate or multivariate. Each component of 
the state vector must be numerical because CFRL involves taking means and standard deviations of the 
state variable.

3. **Action:** The action must be univariate. It should be either categorical or discrete numerical. 
In either case, the actions should be coded as :math:`0, 1, 2, \dots, N_a`, where :math:`N_a` is the 
total number of legit actions. In general, the action does not need to be manually one-hot encoded. 
However, the constructors of the :code:`SequentialPreprocessor` and :code:`SimulatedEnvironment` classes 
provide a :code:`is_action_onehot` argument. If :code:`is_action_onehot=True`, then the action will be 
automatically one-hot encoded internally. 

4. **Reward:** The reward must be univariate and numerical. As in most reinforcement learning settings, 
the reward should be parametrized so that the goal of the agent is to maximize the cumulative 
(discounted) reward.

5. **Environment:** The environment (i.e. underlying Markov decision process) should be modeled as a 
continuing process without any terminal states.

6. **Trajectory Length:** In the data, each individual's trajectory must have the same length. If each 
trajectory is of variable lengths, users might consider truncating each individual's trajectory to 
the same length, but this could be inappropriate in some scenarios.