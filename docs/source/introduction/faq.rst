FAQ
=========================

**1. Does CFRL work for online reinforcement learning?**

    No. the sequential data preprocessing method provided by CFRL is designed for the offline 
    reinforcement learning setting where a pre-collected offline trajectory is available. It 
    does not apply to the online setting in general. 

    Indeed, to our current knowledge, there has yet been a method specifically designed for 
    ensuring counterfactual fairness in online reinforcement learning. Once such a method is 
    developed, we will try to incorporate it into CFRL.