rng = MersenneTwister(1)
using StaticArrays
using ProgressMeter
using Parameters
using JLD
using AutomotiveDrivingModels
using POMDPs
using POMDPToolbox
using GridInterpolations
using DiscreteValueIteration
using LocalApproximationValueIteration
using AutomotiveSensors
using AutomotivePOMDPs
using MDPModelChecking
using DeepQLearning
using DeepRL
using PedCar
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--cost"
        arg_type=Float64
        default=1.0
    "--lr"
        help = "learning rate"
        arg_type = Float64
        default = 1e-4
    "--trace_length"
        help = "number of time steps to train the lstm"
        arg_type = Int64
        default = 40
    "--logdir"
        help = "log directory"
        arg_type = String
        default = "log"
    "--max_steps"
        help = "number of training steps"
        arg_type = Int64
        default = 10000
    "--target_update_freq"
        help = "target network update frequency"
        arg_type = Int64
        default = 2000
    "--eval_freq"
        help = "frequency to run evaluation"
        arg_type = Int64
        default = 10000
    "--n_hidden"
        help = "n fully connected layers before lstm"
        arg_type = Int64
        default = 1
    "--n_nodes"
        arg_type = Int64
        default = 32
    "--lstm_size"
        help = "lstm size"
        arg_type = Int64
        default = 64
    "--gradclip"
        help = "activate gradient clipping"
        action = :store_true
    "--eps_fraction"
        help = "fraction of the training used for decaying epsilon"
        arg_type = Float64
        default = 0.5
    "--eps_end"
        help = "final value of epsilon"
        arg_type = Float64
        default = 0.01
end
parsed_args = parse_args(ARGS, s)

include("../util.jl")
include("../masking.jl")
include("../masked_dqn.jl")

# init mdp
mdp = PedCarMDP(pos_res=2.0, vel_res=2., ped_birth=0.7, car_birth=0.7)
init_transition!(mdp)

# init safe policy
vi_data = load("../pc_processed.jld")
policy = ValueIterationPolicy(mdp, vi_data["qmat"], vi_data["util"], vi_data["pol"])

# init mask
threshold = 0.99
mask = SafetyMask(mdp, policy, threshold)

# init continuous state mdp 
pomdp = UrbanPOMDP(env=mdp.env,
                    # sensor = GaussianSensor(false_positive_rate=0.05, 
                    #                         pos_noise = LinearNoise(min_noise=0.5, increase_rate=0.05), 
                    #                         vel_noise = LinearNoise(min_noise=0.5, increase_rate=0.05)),
                   sensor = PerfectSensor(),
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=1, 
                   car_birth=0.7, 
                   ped_birth=0.7, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   ego_start=20,
                   ΔT=0.5)
pomdp.action_cost = 0.0
pomdp.collision_cost = -parsed_args["cost"]

### Training using DRQN 

fc_in = []
for i=1:parsed_args["n_hidden"]
    push!(fc_in, parsed_args["n_nodes"])
end

solver = DeepRecurrentQLearningSolver(arch=RecurrentQNetworkArchitecture(fc_in=fc_in, lstm_size=parsed_args["lstm_size"]),
                             max_steps = parsed_args["max_steps"],
                             lr = parsed_args["lr"],
                             batch_size = 32,
                             trace_length = parsed_args["trace_length"],
                             target_update_freq = parsed_args["target_update_freq"],
                             buffer_size = 400000,
                             eps_fraction = parsed_args["eps_fraction"],
                             eps_end = parsed_args["eps_end"],
                             train_start = 10000, #10k
                             train_freq = 4,
                             eval_freq = parsed_args["eval_freq"],
                             double_q = true,
                             dueling = true,
                             logdir="drqn-log/"*parsed_args["logdir"],
                             max_episode_length = 100,
                             grad_clip=parsed_args["gradclip"],
                             exploration_policy = masked_linear_epsilon_greedy(parsed_args["max_steps"], parsed_args["eps_fraction"], parsed_args["eps_end"], mask),
                             evaluation_policy = masked_evaluation(mask),
                             verbose = true,
                             rng = rng
                            )

env = POMDPEnvironment(pomdp)
policy = solve(solver, env)
# save weights!
DeepQLearning.save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")
