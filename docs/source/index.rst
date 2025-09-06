.. CFRL documentation master file, created by
   sphinx-quickstart on Sat Jun 14 15:17:14 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CFRL documentation
==================

Welcome to CFRL, a Python library for counterfactually fair reinforcement learning! 
The acronym "CFRL" stands for "Counterfactual Fairness in Reinforcement Learning". CFRL provides 
algorithms that ensure counterfactual fairness in reinforcement learning and builds tools for 
evaluating the value and counterfactual fairness of reinforcement learning policies. 

To install CFRL, run 

.. code-block:: bash
   
   $ pip install cfrl

This project is still being perfected. We will continue adding new functionalities and expanding 
the use cases of CFRL. We appreciate your patience and support!

`[CFRL Github repository] <https://github.com/JianhanZhang/CFRL>`_

[CFRL software paper]

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   introduction/getting_started
   introduction/computing_times
   introduction/faq

.. toctree::
   :maxdepth: 2
   :caption: Inputs and Outputs

   inputs_and_outputs/data_requirements
   inputs_and_outputs/trajectory_arrays
   inputs_and_outputs/tabular_trajectory_data

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/example_workflows
   tutorials/common_issues

.. toctree::
   :maxdepth: 2
   :caption: Interface

   interface/index

.. toctree::
   :maxdepth: 2
   :caption: Customizations

   customizations/custom_preprocessors
   customizations/custom_agents

.. toctree::
   :maxdepth: 2
   :caption: About CFRL

   about_cfrl/the_cfrl_team