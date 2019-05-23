# Safe Planning Under Uncertainty for Autonomous Driving

Experiments on safe planning under uncertainty for autonomous driving. 
This code base combines reachability analysis, reinforcement learning, and decomposition methods to compute safe and efficient policies for autonomous vehicles.

**Reference**: M. Bouton, A. Nakhaei, K. Fujimura, and M. J. Kochenderfer, “Safe reinforcement learning with scene decomposition for navigating complex urban environments,” in IEEE Intelligent Vehicles Symposium (IV), 2019. 

## Installation

To install all the dependencies, run the following in the Julia REPL:

```julia 
using Pkg
Pkg.add(PackageSpec(url="https://github.com/sisl/Vec.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/Records.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveDrivingModels.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutoViz.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutoUrban.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveSensors.jl"))
Pkg.add(PackageSpec(url="https://github.com/JuliaPOMDP/RLInterface.jl"))
Pkg.add(PackageSpec(url="https://github.com/JuliaPOMDP/DeepQLearning.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotivePOMDPs.jl"))
Pkg.add(PackageSpec(url="https://github.com/MaximeBouton/POMDPModelChecking.jl"))
``` 

## Folder structure

- `src/` contains the implementation of the safe RL policy and the decomposition method.
- `RNNFiltering/` contains data_generation and training_script for the ensemble RNN belief updater
- `training_scripts/` contains training scripts for the safe RL and RL policies
- `evaluation/` contains evaluation scripts to evaluate RL, safe RL, and baseline policies.
- `notebooks/` contains jupyter notebook for visualization and debugging. 


## Code to run

To visualize any of the policy use `notebooks/interactive_evaluation.ipynb`

For a detailed description of the evaluation scenarios run `notebooks/evaluation_scenarios.ipynb`

Other notebooks are used for prototyping and debugging. 


## Main Dependencies

- AutomotivePOMDPs.jl contains all the driving scenario and MDP models
- POMDPModelChecking.jl 
- DeepQLearning.jl (Flux.jl backend)
