# CFRL: A Python Library for Counterfactually Fair Reinforcement Learning

Documentation: To be updated.

Paper: To be updated. 

## Installation

```bash 
$ pip install CFRL
```

## A Brief Introduction to Counterfactual Fairness

Counterfactual fairness is one variation of fairness metrics. However, unlike well-known group 
fairness metrics such as demographic parity and equal opportunity, counterfactual fairness defines 
fairness based on causal reasoning and enforces it at the individual level. In short, a reinforcement learning policy is counterfactually fair if, at every time step, it would assign the same decisions 
with the same probabilities for an individual had the individual belong to a different subgroup 
defined by some sensitive attribute (such as race and gender). At its core, counterfactual fairness 
views the observed states and rewards as biased proxies of the (unobserved) true underlying states 
and rewards, where the bias is associated (?) with the sensitive attribute. Thus, to ensure 
counterfactual fairness, we want the policy to be based on the true underlying states and rewards 
rather than their biased proxies. 

We refer interested readers to [Kusner et al. (2017)](https://arxiv.org/abs/1703.06856) for a detailed discussion of counterfactual fairness in the single-stage prediction setting, and to 
[Wang et al. (2025)](https://arxiv.org/abs/2501.06366) for a detailed discussion of counterfactual 
fairness in the reinforcement learning setting.

## Key Functionalities

CFRL is designed with two main functionalities: 

1. Provide algorithms that enforce counterfactual fairness for reinforcement learning policies. 
The current version of CFRL implements the sequential data preprocessing algorithm proposed by
[Wang et al. (2025)](https://arxiv.org/abs/2501.06366) for offline reinforcement learning. The 
algorithm takes in an offline RL trajectory and outputs a preprocessed, bias-free trajectory. The 
preprocessed trajectory can then be passed to any existing offline reinforcement learning algorithms 
for training, and the learned policy should be approximately counterfactually fair. 

2. Provides a platform to evaluate RL policies based on counterfactual fairness. After passing in 
their policy and a trajectory dataset from the target environment, users can assess how well their 
policies perform in the target environment in terms of the discounted cumulative reward and 
counterfactual fairness metric.

## High-level Design
| Module         | Functionalities                                                                                                                                                                                                                                                                                                |
|------------|------------------------------------------------------------|
| `reader`       | Implements functions that read tabular trajectory data from either a `.csv` file or a `pandas.Dataframe`. Also implements functions that export trajectory data to either a `.csv` file or a `pandas.Dataframe`.                                                                                               |
| `preprocessor` | Implements the data preprocessing algorithm introduced in Wang et al. (2025).                                                                                                                                                                                                                                  |
| `agents`       | Implements a fitted Q-iteration (FQI) algorithm, which learns RL policies and makes decisions based on the learned policy. Users can also pass a preprocessor to the FQI; in this case, the FQI will be able to take in unpreprocessed trajectories and directly output counterfactually fair policies.        |
| `environment`  | Implements a synthetic environment that produces synthetic data as well as a simulated environment that simulates the transition dynamics of the environment underlying some real-world RL trajectory data. Also implements functions for sampling trajectories from the synthetic and simulated environments. |
| `fqe`          | Implements a fitted Q-evaluation (FQE) algorithm, which can be used to evaluate the value of a trained policy.                                                                                                                                                                                                 |
| `evaluation`   | Implements functions that evaluate the value and fairness of a trained policy. Depending on the user's needs, the evaluation can be done either in a synthetic environment or in a simulated environment.                                                                                                      |

![Workflow Chart](./supps/workflow%20chart%20cropped.PNG)

A general package workflow is as follows: First, simulate a trajectory using `environment` or read in a trajectory using `reader`. Then, train a preprocessor using `preprocessor` to remove the bias in the trajectory data. After that, pass the preprocessed trajectory into the FQI algorithm in `agents` to learn a counterfactually fair policy. Finally, use functions in `evaluation` to evaluate the value and fairness of the trained policy. See ... for more detailed workflow examples.

## Examples

This section provide a few short examples showcasing some use cases of CFRL.

### Data Preprocessing

To be updated. 

### Assessing Algorithms using Simulation

To be updated. 

### Assessing Policies using Real Data

To be updated. 

## Testing
We provide unit tests as well as integration tests for the main functions of the CFRL. The tests can be 
found in the `tests` folder of the `CFRL` repository. To run the test, first install `pytest`: 

<pre> ```bash $ pip install pytest``` </pre>

Then run 

<pre> ```bash $ python -m pytest``` </pre>