# Safe Planning Under Uncertainty via Formal Methods for Autonomous Driving

Experiments on safe planning under uncertainty for autonomous driving. 
This code base combines reachability analysis and reinforcement learning to compute safe and efficient policies for autonomous vehicles

## Code to run

- `pedmdp_script.jl` compute the safety mask, run some evaluations and train a policy in the constrained action space for a scenario involving the ego vehicle and one pedestrian to avoid
- `carmdp_script.jl` compute the safety mask, run some evaluations and train a policy in the constrained action space for a scenario involving the ego vehicle and one car to avoid
- `pedmdp_training.jl` run DQN to train a policy (no prior reachability analysis), and evaluate the policy for a scenario involving the ego vehicle and one pedestrian to avoid
- `carmdp_training.jl` run DQN to train a policy (no prior reachability analysis), and evaluate the policy for a scenario involving the ego vehicle and one car to avoid
- `jointmdp_script.jl` load the safety masks for the car problem and the pedestrian problem, combine the two mask and train a policy in the constrained action space for a scenario involving the ego vehicle, one car and one pedestrian. 
- `jointmdp_training.jl` run DQN to train a policy  (no prior reachability analysis), and evaluate the policy for a scenario involving the ego vehicle, one car and one pedestrian

The following notebooks allows to interact with the code and visualize the trained policies and the safety mask: `ped_mdp.ipynb`, `car_mdp.ipynb`, `joint_problem.ipynb`.

- `gridworld/gridworld.ipynb` contains an illustrative example of the training procedure on a gridworld problem


**/!\ the models in `mdp_models/` are deprecated and have been moved to AutomotivePOMDPs.jl`**

## Dependencies

- AutomotivePOMDPs.jl contains all the driving scenario and MDP models
- MDPModelChecking.jl (storm model checker backend)
- DeepQLearning.jl (tensorflow backend)

