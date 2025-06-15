.. _trajectory_arrays:

Trajectory Arrays
===============================

In CFRL, a trajectory refers to the set of collected observed tuples 
:math:`\{(z_i, s_{i0}, a_{i0}, r_{i0}, s_{i1}, \dots, s_{i,T-1}, a_{i,T-1}, r_{i,T-1}, s_{iT}): i=1,\dots,N\}` 
describing the sensitive attribute, state, action, and reward of each individual (or subject) at each 
time step. Each :math:`t=0,\dots,T` is called a time step, and the observed tuple 
:math:`(s_{it}, a_{it}, r_{it}, s_{i,t+1})` is called the transition for individual :math:`i` at time 
step :math:`t`. Let :math:`N` be the total number of individuals and :math:`T` be the total number of 
transitions in the trajectory.

This section introduces Trajectory Arrays, which is how trajectories are represented in CFRL. 
Any trajectory satisfying the :ref:`data requirements <data_requirements>` can be represented by 
Trajectory Arrays. The trajectory inputs and outputs of CFRL functions and classes are all in the 
form of Trajectory Arrays. To convert trajectory data from a tabular format to Trajectory Arrays or 
from Trajectory Arrays to a tabular data, see :ref:`Tabular Trajectory Data <tabular_trajectory_data>`.

The sensitive attributes, states, actions, and rewards in a trajectory are represented by 
Trajectory Arrays of different formats, which are introduced below.

Sensitive Attributes Format
--------------------------------

A Trajectory Array in the Sensitive Attribute Format is used to store the observed sensitive 
attributes of each individual in the trajectory. It is a 2D list or array with shape :code:`(N, zdim)` 
where :code:`zdim` is the number of components in the sensitive attribute vector. The (i, j)-th 
entry of the list or array represents the j-th component of the observed sensitive attribute of the 
i-th individual. Note that if the sensitive attribute is univariate, then a Trajectory Array in the 
Sensitive Attribute Format should have shape :code:`(N, 1)` rather than :code:`(N,)`.

For example, consider a trajectory dataset with 3 individuals where the sensitive attribute is 
bivariate. Then the sensitive attributes of this trajectory can be represented in the Sensitive 
Attribute Format as

+---------------+---------------+
| :math:`z_1^1` | :math:`z_1^2` |
+---------------+---------------+
| :math:`z_2^1` | :math:`z_2^2` |
+---------------+---------------+
| :math:`z_3^1` | :math:`z_3^2` |
+---------------+---------------+

Single-time States Format
--------------------------------

A Trajectory Array in the Single-time States Attribute Format is used to store the state
of each individual in the trajectory at a single time step. It is a 2D list or array with 
shape :code:`(N, xdim)` where :code:`xdim` is the number of components in the state vector. The 
(i, j)-th entry of the list or array represents the j-th component of the state variable of the 
i-th individual at the given time step. Note that if the state vector is univariate, then a 
Trajectory Array in the Single-time States Format should have shape :code:`(N, 1)` rather than 
:code:`(N,)`.

For example, consider a trajectory dataset with 3 individuals where the state variable is 
bivariate. Then the states of this trajectory at some time step :math:`t` can be represented in the 
Single-time States Format as

+------------------+------------------+
| :math:`x_{1t}^1` | :math:`x_{1t}^2` |
+------------------+------------------+
| :math:`x_{2t}^1` | :math:`x_{2t}^2` |
+------------------+------------------+
| :math:`x_{3t}^1` | :math:`x_{3t}^2` |
+------------------+------------------+

Full-trajectory States Format
--------------------------------

A Trajectory Array in the Full-trajectory States Format is used to store the state 
of each individual in the trajectory at all time steps. It is a 3D list or array with 
shape :code:`(N, T+1, xdim)` where :code:`xdim` is the number of components in the state vector. The 
(i, j, k)-th entry of the list or array represents the k-th component of the state variable of the 
i-th individual at the j-th time step. Note that if the state vector is univariate, then a Trajectory 
Array in the Single-time States Format should have shape :code:`(N, T+1, 1)` rather than :code:`(N, T+1)`.

For example, consider a trajectory dataset with 3 individuals and 3 transitions where the state 
variable is bivariate. Then the states of this trajectory at all time steps can be represented in the 
Full-trajectory States Format as

+------------------------------+------------------------------+------------------------------+------------------------------+
| :math:`[x_{10}^1, x_{10}^2]` | :math:`[x_{11}^1, x_{11}^2]` | :math:`[x_{12}^1, x_{12}^2]` | :math:`[x_{13}^1, x_{13}^2]` |
+------------------------------+------------------------------+------------------------------+------------------------------+
| :math:`[x_{20}^1, x_{20}^2]` | :math:`[x_{21}^1, x_{21}^2]` | :math:`[x_{22}^1, x_{22}^2]` | :math:`[x_{23}^1, x_{23}^2]` |
+------------------------------+------------------------------+------------------------------+------------------------------+
| :math:`[x_{30}^1, x_{30}^2]` | :math:`[x_{31}^1, x_{31}^2]` | :math:`[x_{32}^1, x_{32}^2]` | :math:`[x_{33}^1, x_{33}^2]` |
+------------------------------+------------------------------+------------------------------+------------------------------+

Single-time Actions Format
--------------------------------

A Trajectory Array in the Single-time Actions Attribute Format is used to store the action 
of each individual in the trajectory at a single time step. It is a 1D list or array with 
shape :code:`(N,)`. The i-th entry of the list or array represents action of the i-th individual at 
the given time step.

For example, consider a trajectory dataset with 3 individuals. Then the actions of this trajectory at 
some time step :math:`t` can be represented in the Single-time Actions Format as

+----------------+----------------+----------------+
| :math:`a_{1t}` | :math:`a_{2t}` | :math:`a_{3t}` |
+----------------+----------------+----------------+

Full-trajectory Actions Format
--------------------------------

A Trajectory Array in the Full-trajectory Actions Format is used to store the action 
of each individual in the trajectory at all time steps. It is a 2D list or array with 
shape :code:`(N, T)`. The (i, j)-th entry of the list or array represents the action of the 
i-th individual at the j-th time step. 

For example, consider a trajectory dataset with 3 individuals and 3 transitions. Then the actions 
of this trajectory at all time steps can be represented in the Full-trajectory Actions Format as

+----------------+----------------+----------------+
| :math:`a_{10}` | :math:`a_{11}` | :math:`a_{12}` |
+----------------+----------------+----------------+
| :math:`a_{20}` | :math:`a_{21}` | :math:`a_{22}` |
+----------------+----------------+----------------+
| :math:`a_{30}` | :math:`a_{31}` | :math:`a_{32}` |
+----------------+----------------+----------------+

Single-time Rewards Format
--------------------------------

A Trajectory Array in the Single-time Rewards Attribute Format is used to store the reward 
of each individual in the trajectory at a single time step. It is a 1D list or array with 
shape :code:`(N,)`. The i-th entry of the list or array represents reward of the i-th individual at 
the given time step.

For example, consider a trajectory dataset with 3 individuals. Then the rewards of this trajectory at 
some time step :math:`t` can be represented in the Single-time Rewards Format as

+----------------+----------------+----------------+
| :math:`r_{1t}` | :math:`r_{2t}` | :math:`r_{3t}` |
+----------------+----------------+----------------+

Full-trajectory Rewards Format
--------------------------------

A Trajectory Array in the Full-trajectory Rewards Format is used to store the reward 
of each individual in the trajectory at all time steps. It is a 2D list or array with 
shape :code:`(N, T)`. The (i, j)-th entry of the list or array represents the reward of the 
i-th individual at the j-th time step. 

For example, consider a trajectory dataset with 3 individuals and 3 transitions. Then the actions 
of this trajectory at all time steps can be represented in the Full-trajectory Rewards Format as

+----------------+----------------+----------------+
| :math:`r_{10}` | :math:`r_{11}` | :math:`r_{12}` |
+----------------+----------------+----------------+
| :math:`r_{20}` | :math:`r_{21}` | :math:`r_{22}` |
+----------------+----------------+----------------+
| :math:`r_{30}` | :math:`r_{31}` | :math:`r_{32}` |
+----------------+----------------+----------------+