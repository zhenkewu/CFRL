---
title: "CFRL: A Python library for counterfactually fair offline reinforcement learning via sequential data preprocessing"
tags:
  - counterfactual fairness
  - reinforcement learning
  - CFRL
  - Python
authors:
  - name: Jianhan Zhang
    corresponding: false 
    affiliation: 1
  - name: Jitao Wang
    corresponding: false 
    affiliation: 2
  - name: Chengchun Shi
    corresponding: false 
    affiliation: 3
  - name: John D. Piette
    corresponding: false 
    affiliation: 4
  - name: Donglin Zeng
    corresponding: false 
    affiliation: 2
  - name: Zhenke Wu
    corresponding: true 
    affiliation: 2

affiliations:
 - name: 'Department of Statistics, University of Michigan, USA'
   index: 1
 - name: 'Department of Biostatistics, University of Michigan, USA'
   index: 2
 - name: 'Department of Statistics, London School of Economics, UK'
   index: 3
 - name: 'Department of Health Behavior and Health Equity, School of Public Health, University of Michigan, USA'
   index: 4

citation_author: Zhang et. al.
date: 9 August 2025
year: 2025
journal: JOSS
bibliography: paper.bib
preamble: >
  \usepackage{longtable}
  \usepackage{makecell}
  \usepackage{tabularx}
  \usepackage{hyperref}
  \usepackage{graphicx}
  \usepackage{amsmath}
  \usepackage{booktabs}
  \usepackage{amsfonts}
  \usepackage{tabulary}
  \usepackage{ragged2e}
  \usepackage{floatrow}
  \floatsetup[table]{capposition=top}
  \floatsetup[figure]{capposition=top}
---

# Summary

Reinforcement learning (RL) aims to learn and evaluate a sequential
decision-making rule, often referred to as a “policy”, that maximizes
expected discounted cumulative rewards to optimize population-level benefit in an environment across possibly infinitely many time steps. RL has gained popularity in fields such as healthcare, banking, autonomous driving, and more recently large language model pre-training. However, the sequential decisions made by an RL algorithm may disadvantage individuals with certain values of a sensitive attribute (e.g., race/ethnicity, income, gender, education level). An RL algorithm learns an optimal policy that makes decisions based on observed state variables. If certain values of the sensitive attribute influence the state variables in a way that influences the policy, such as by leading the policy to systematically withhold services from an individual, unfairness will result. For example, Hispanics may under-report their pain levels due to cultural factors, misleading a fairness-blind RL agent to assign less therapist time to them [@piette2023powerED]. Deployment of RL algorithms without careful fairness considerations can raise concerns and erode public trust in high-stake settings.

To formally define and address the fairness problem in sequential decision making settings, @wang2025cfrl extended the concept of single-stage counterfactual fairness (CF) in a structural causal framework [@kusner2018cf] to the
multi-stage setting and proposed a data preprocessing algorithm that
ensures CF. A policy is counterfactually fair if, at every time step, the probability of assigning any action does not change had the individual's sensitive attribute taken a different value, while holding constant other historical exogenous variables and actions. In this light, the data preprocessing algorithm ensures CF by constructing new state variables that are not impacted by the sensitive attribute(s). The reward values in the data are also preprocessed, but the purpose of preprocessing the rewards is to improve the value of the learned optimal policy rather than ensure CF. We refer interested readers to @wang2025cfrl for more technical details.

The `CFRL` library implements the data preprocessing algorithm proposed by @wang2025cfrl and provides a suite of tools to evaluate the value and counterfactual unfairness level achieved by 
any given policy. Here, "CFRL" stands for "Counterfactual Fairness in Reinforcement Learning". The library produces preprocessed trajectories that can be used by
an off-the-shelf offline RL algorithm, such as fitted Q-iteration (FQI) [@riedmiller2005fqi], to learn an optimal CF
policy. The library can also simply read in any policy following a required format and return its
value and counterfactual unfairness level in the environment of interest, where the environment can be either pre-specified or learned from the data.

# Statement of Need

Many existing Python libraries implement algorithms designed to ensure fairness
in machine learning. For example, `Fairlearn` [@weerts2023fairlearn] and 
`aif360` [@aif360-oct-2018] provide tools 
for mitigating bias in single-stage machine learning predictions under
statistical association-based fairness criteria such as demographic
parity and equal opportunity. However, existing libraries do not focus on 
counterfactual fairness, which defines an individual-level fairness concept from a causal
perspective, and they cannot be easily extended to the generalized reinforcement
learning setting. Scripts available from `ml-fairness-gym` [@fairness_gym] allow users 
to simulate unfairness in sequential decision-making, but they neither 
implement algorithms that reduce unfairness nor address CF. To our knowledge, @wang2025cfrl is the first 
work to study CF in RL. 
Correspondingly, `CFRL` is also the first code library to address CF in the RL setting.

The contribution of CFRL is two-fold. First, `CFRL` implements a data
preprocessing algorithm that ensures CF in offline RL.
For each individual in the data, the preprocessing
algorithm sequentially estimates the counterfactual states under different sensitive
attribute values and concatenates all of the individual’s counterfactual
states at each time point into a new state vector. The preprocessed data can then be
directly used by existing RL algorithms for policy learning, and the
learned policy should be counterfactually fair up to finite-sample estimation accuracy. Second, `CFRL`
provides a platform for assessing RL policies based on CF. After passing in any policy and a data trajectory from the
environment of interest, users can estimate the value and counterfactual unfairness level achieved by the policy in the environment of interest. 

# High-level Design

The `CFRL` library is composed of 5 major modules. The functionalities
of the modules are summarized in the table below.

+--------------+--------------------------------------------------------------------------------------+
|Module        |Functionalities                                                                       |
+==============+======================================================================================+
|`reader`      |Implements functions that read tabular trajectory data from either a `.csv` file or a |
|              |`pandas.Dataframe` into an array format required by `CFRL`. Also implements functions |
|              |that export trajectory data to either a `.csv` file or a `pandas.Dataframe`.          |
+--------------+--------------------------------------------------------------------------------------+
|`preprocessor`|Implements the data preprocessing algorithm introduced in @wang2025cfrl.              |
+--------------+--------------------------------------------------------------------------------------+
|`agents`      |Implements a fitted Q-iteration (FQI) algorithm [@riedmiller2005fqi], which learns RL |
|              |policies and makes                                                                    |
|              |decisions based on the learned policy. Users can also pass a preprocessor to the FQI; | 
|              |in this case, the FQI will be able to take in unpreprocessed trajectories, internally | 
|              |preprocess the input trajectories, and directly output counterfactually fair policies.|
+--------------+--------------------------------------------------------------------------------------+
|`environment` |Implements a synthetic environment that produces synthetic data as well as a simulated|
|              |environment that estimates and simulates the transition dynamics of the unknown       | 
|              |environment underlying some                                                           |
|              |real-world RL trajectory data. Also implements functions for sampling trajectories    |
|              |from the synthetic and simulated environments.                                        |
+--------------+--------------------------------------------------------------------------------------+
|`evaluation`  |Implements functions that evaluate the value and counterfactual unfairness level of a |
|              |policy. Depending on the user's needs, the evaluation can be done either in a         | 
|              |synthetic environment or in a simulated environment.                                  |
+==============+======================================================================================+

A general CFRL workflow is as follows: First, simulate a trajectory using `environment` or read 
in a trajectory using `reader`. Then, train a preprocessor using `preprocessor` and preprocess the training trajectory data. After that, pass the preprocessed trajectory into the FQI algorithm in 
`agents` to learn a counterfactually fair policy. Finally, use functions in `evaluation` to 
evaluate the value and counterfactual unfairness level of the trained policy. 

# Data Example

We provide a data example to demonstrate how `CFRL` learns a counterfactually fair policy from real-world trajectory data with unknown underlying transition dynamics. We also show how `CFRL` evaluates the value and counterfactual unfairness level of the learned policy. We note that this is only one of the many workflows that `CFRL` can perform. For example, `CFRL` can also generate synthetic trajectory data and use it to evaluate the value and counterfactual unfairness level resulting from some custom data preprocessing methods. We refer interested readers to the ["Example Workflows"](https://cfrl-documentation.netlify.app/tutorials/example_workflows) section of the CFRL documentation for more workflow examples.

#### Load Data

In this demonstration, we use an offline trajectory generated from a `SyntheticEnvironment` following some pre-specified transition rules. Although the data is actually synthesized, we treat it as if it is from some unknown environment for pedagogical convenience.

The trajectory contains 500 individuals (i.e. $N=500$) and 10 transitions (i.e. $T=10$). The sensitive attribute variable and the state variable are both univariate. The sensitive attribute is binary ($0$ or $1$). The actions are also binary ($0$ or $1$) and were sampled using a behavior policy that selects $0$ or $1$ randomly with equal probability. The trajectory is stored in a tabular format in a `.csv` file. We use `read_trajectory_from_csv()` to load the trajectory from the `.csv` format into the array format required by `CFRL`.
```python
zs, states, actions, rewards, ids = read_trajectory_from_csv(
    path='../data/sample_data_large_uni.csv', z_labels=['z1'], 
    state_labels=['state1'], action_label='action', reward_label='reward', 
    id_label='ID', T=10)
```

We then split the trajectory data into a training set (80%) and a testing set (20%) using scikit-learn's `train_test_split()`. The training set is used to train the counterfactually fair policy, while the testing set is used to evaluate the value and counterfactual unfairness level achieved by the policy.

```python
(zs_train, zs_test, states_train, states_test, 
 actions_train, actions_test, rewards_train, rewards_test
) = train_test_split(zs, states, actions, rewards, test_size=0.2)
```

#### Train Preprocessor & Preprocess Trajectories

We now train a `SequentialPreprocessor` and preprocess the trajectory. The `SequentialPreprocessor` ensures the learned policy is counterfactually fair by constructing new state variables that are not impacted by the sensitive attribute. Due to limited trajectory data, the data to be preprocessed will also be the data used to train the preprocessor, so we set `cross_folds=5` to reduce overfitting. In this case, `train_preprocessor()` will internally divide the training data into 5 folds, and each fold is preprocessed using a model that is trained on the other 4 folds. We initialize the `SequentialPreprocessor`, and `train_preprocessor()` will take care of both preprocessor training and trajectory preprocessing.

```python
sp = SequentialPreprocessor(z_space=[[0], [1]], num_actions=2, cross_folds=5, 
                            mode='single', reg_model='nn')
states_tilde, rewards_tilde = sp.train_preprocessor(
    zs=zs_train, xs=states_train, actions=actions_train, rewards=rewards_train)
```

#### Counterfactually Fair Policy Learning

Next we train a counterfactually fair policy using the preprocessed data and `FQI` with `sp` as its internal preprocessor. By default, the input data will first be preprocessed by `sp` before being used for policy learning. However, since the training data `state_tilde` and `rewards_tilde` are already preprocessed in our case, we set `preprocess=False` during training so that the input trajectory will not be preprocessed again by the internal preprocessor (i.e. `sp`).

```python
agent = FQI(num_actions=2, model_type='nn', preprocessor=sp)
agent.train(zs=zs_train, xs=states_tilde, actions=actions_train, 
            rewards=rewards_tilde, max_iter=100, preprocess=False)
```

#### `SimulatedEnvironment` Training

Before moving on to the evaluation stage, there is one more step: We need to train a `SimulatedEnvironment` that mimics the transition rules of the true environment that generated the training trajectory, which will be used by the evaluation functions to simulate the true data-generating environment. To do so, we initialize a `SimulatedEnvironment` and train it on the whole trajectory data (i.e. training set and testing set combined).

```python
env = SimulatedEnvironment(num_actions=2, state_model_type='nn', 
                           reward_model_type='nn')
env.fit(zs=zs, states=states, actions=actions, rewards=rewards)
```

#### Value and Counterfactual Unfairness Level Evaluation

We now use `evaluate_value_through_fqe()` and `evaluate_fairness_through_model()` to estimate the value and counterfactual unfairness level achieved by the trained policy when interacting with the environment of interest, respectively. The counterfactual unfairness level is represented by a metric from 0 to 1, with 0 representing perfect fairness and 1 indicating complete unfairness. We use the testing set for evaluation.

```python
value = evaluate_reward_through_fqe(zs=zs_test, states=states_test, 
    actions=actions_test, rewards=rewards_test, policy=agent, model_type='nn')
cf_metric = evaluate_fairness_through_model(env=env, zs=zs_test, states=states_test, 
                                            actions=actions_test, policy=agent)
```

The estimated value is $7.358$ and CF metric is $0.042$, which indicates our policy is close to being perfectly counterfactually fair. Indeed, the CF metric should be exactly 0 if we know the true dynamics of the environment of interest; the reason why it is not exactly 0 here is because we need to estimate the dynamics of the environment of interest during preprocessing, which can introduce finite-sample errors.

#### Comparisons Against Baseline Methods

We can compare the sequential data preprocessing method in `CFRL` against a few baselines: Random, which selects each action randomly with equal probability; Full, which uses all variables, including the sensitive attribute, for policy learning; and Unaware, which uses all variables except the sensitive attribute for policy learning. We implemented these baselines and evaluated their values and counterfactual unfairness levels as part of the code example of the "Assessing Policies Using Real Data" workflow in the ["Example Workflows"](https://cfrl-documentation.netlify.app/tutorials/example_workflows) section of the CFRL documentation. We summarize below the values and CF metrics calculated in this code example, where "ours" stands for the `SequentialPreprocessor`.

+-------------------------------+--------+-------+-------+-------+
|                               |Random  |Full   |Unaware|Ours   |                                                                
+===============================+========+=======+=======+=======+
|Value                          |$-1.444$|$8.606$|$8.588$|$7.358$|
+-------------------------------+--------+-------+-------+-------+
|Counterfactual Unfairness Level|$0$     |$0.407$|$0.446$|$0.042$|
+===============================+========+=======+=======+=======+

By definition, the "random" baseline always achieves perfect CF. On the other hand, "ours" resulted in much fairer policies than "full" and "unaware", which suggests that the `SequentialPreprocessor` can effectively improve CF. Nevertheless, as a trade-off for higher CF, "ours" achieved a lower value than "full" and "unaware".

# Non-convergence in `CFRL`

We remark that non-convergence might arise in `CFRL` in `evaluate_reward_through_fqe()` and during the training of `SequentialPreprocessor`, `SimulatedEnvironment`, and `FQI`. In this case, `CFRL` provides a tool that can check for potential non-convergence using loss monitoring. More discussions about the sources, checks, and fixes of non-convergence in `CFRL` can be found in the "Common Issues" section of the CFRL documentation.

# Conclusions

`CFRL` is a Python library that enables CF reinforcement
learning through data preprocessing. It also provides tools to calculate
the value and unfairness level of a given policy. To our knowledge, it is the first library to address CF
problems in the context of RL. The practical utility of `CFRL` can be further improved via extensions. First, the current
`CFRL` implementation requires every individual in the offline dataset to
have the same number of time steps. Extending the library to accommodate
variable-length episodes can improve its flexibility and usefulness.
Second, `CFRL` can further combine the
preprocessor with popular offline RL algorithm libraries such as
`d3rlpy` [@d3rlpy], or connect the evaluation functions with established RL
environment libraries such as `gym` [@towers2024gymnasium]. Third, generalization to
non-additive counterfactual states reconstruction can make `CFRL` theoretically more versatile. We leave these extensions 
to future updates.

<!-- # Acknowledgements

This is the acknowledgements. -->

# References
