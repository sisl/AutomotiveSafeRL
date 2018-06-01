rng = MersenneTwister(1)
using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DiscreteValueIteration, DeepQLearning, DeepRL
using ProgressMeter, Parameters, JLD

include("masking.jl")
include("util.jl")
include("render_helpers.jl")
include("masked_dqn.jl")

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params)

pomdp = UrbanPOMDP(env=env,
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=1, 
                   car_birth=0.3, 
                   ped_birth=0.7, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   pos_obs_noise = 0., # fully observable
                   vel_obs_noise = 0.)


### Training


#### Training using DQN in high fidelity environment

max_steps = 750000
eps_fraction = 0.5 
eps_end = 0.01 
solver = DeepQLearningSolver(max_steps = max_steps, eps_fraction = eps_fraction, eps_end = eps_end,
                       lr = 0.0001,                    
                       batch_size = 32,
                       target_update_freq = 5000,
                       max_episode_length = 100,
                       train_start = 40000,
                       buffer_size = 400000,
                       eval_freq = 30000,
                       arch = QNetworkArchitecture(conv=[], fc=[32,32,32]),
                       double_q = true,
                       dueling = true,
                       prioritized_replay = true,
                       verbose = true,
                       logdir = "jointmdp-log/log_nm4",
                       rng = rng)


env = POMDPEnvironment(pomdp)
policy = solve(solver, env)
save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")
# evaluate resulting policy
println("\n EVALUATE TRAINED POLICY \n")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, policy, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)

# expected results
# Summary for 10000 episodes:
# Average reward: 0.308
# Average # of steps: 23.303
# Average # of violations: 8.870


# Summary for 10000 episodes:
# Average reward: 0.286
# Average # of steps: 34.757
# Average # of violations: 6.190