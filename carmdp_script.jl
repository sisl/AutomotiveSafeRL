rng = MersenneTwister(1)
using AutomotivePOMDPs
using POMDPStorm
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DeepQLearning, DeepRL
using DiscreteValueIteration
using ProgressMeter, Parameters, JLD

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [10.0, 10., 10.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

               
include("mdp_models/discretization.jl")
include("mdp_models/interpolation.jl")
include("mdp_models/car_mdp/pomdp_types.jl")
include("mdp_models/car_mdp/state_space.jl")
include("mdp_models/car_mdp/transition.jl")
include("mdp_models/car_mdp/render_helpers.jl")

mdp = CarMDP(env = env, vel_res=1.0, pos_res=2.0);
labels = labeling(mdp);
@printf("\n")
@printf("spatial resolution %2.1f m \n", mdp.pos_res)
@printf("car velocity resolution %2.1f m \n", mdp.vel_res)
@printf("number of states %d \n", n_states(mdp))
@printf("number of actions %d \n", n_actions(mdp))
@printf("\n")

property = "Pmax=? [ (!\"crash\") U \"goal\"]" 
threshold = 0.9999
@printf("Spec: %s \n", property)
@printf("Threshold: %f \n", threshold)

println("Model checking...")
result = model_checking(mdp, labels, property, transition_file_name="carmdp.tra", labels_file_name="carmdp.lab")

P = get_proba(mdp, result);

#TODO replace by a do block
mask = nothing # declare
if isfile("carmask.jld")
    println("Loading safety mask from carmask.jld")
    mask_data = JLD.load("carmask.jld")
    mask = mask_data["mask"]
    @printf("Mask threshold %f", mask.threshold)
else
    println("Computing Safety Mask...")
    mask = SafetyMask(mdp, result, threshold);
    JLD.save("carmask.jld", "mask", mask)
    println("Mask saved.")
end

### EVALUATE MASK 
rand_pol = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng);


function evaluation_loop(mdp::MDP, policy::Policy; n_ep::Int64 = 1000, max_steps::Int64 = 500, rng::AbstractRNG = Base.GLOBAL_RNG)
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    d0 = initial_state_distribution(mdp)
    for ep=1:n_ep
        s0 = rand(rng, d0)
        hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        hist = simulate(hr, mdp, policy, s0)
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        violations[ep] = sum(hist.reward_hist .< 0.) #+ Int(n_steps(hist) >= max_steps)
    end
    return rewards, steps, violations
end

function print_summary(rewards, steps, violations)
    @printf("Summary for %d episodes: \n", length(rewards))
    @printf("Average reward: %.3f \n", mean(rewards))
    @printf("Average # of steps: %.3f \n", mean(steps))
    @printf("Average # of violations: %.3f \n", mean(violations)*100)
end

@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, rand_pol, n_ep=1000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)

# vi_policy = nothing # declare
# if isfile("car_vi_policy.jld")
#     policy_data = JLD.load("car_vi_policy.jl")
#     vi_policy = policy_data["policy"]
# else
#     println("Running Value Iteration...")
#     solver = ValueIterationSolver(max_iterations=1000)
#     mdp.collision_cost = -1
#     vi_policy = solve(solver, mdp, verbose=true)
# end

# JLD.save("car_vi_policy.jld", "policy", vi_policy)
# println("Policy saved.")


### EVALUATE IN HIGH FIDELITY ENVIRONMENT

pomdp = UrbanPOMDP(env=env,
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=0, 
                   car_birth=0.3, 
                   ped_birth=0.3, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   pos_obs_noise = 0., # fully observable
                   vel_obs_noise = 0.);
umdp = UnderlyingMDP(pomdp);

function evaluation_loop(pomdp::POMDP, policy::Policy; n_ep::Int64 = 1000, max_steps::Int64 = 500, rng::AbstractRNG = Base.GLOBAL_RNG)
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    up = FastPreviousObservationUpdater{obs_type(pomdp)}()
    for ep=1:n_ep
        s0 = initial_state(pomdp, rng)
        o0 = generate_o(pomdp, s0, rng)
        b0 = initialize_belief(up, o0)
        hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        hist = simulate(hr, pomdp, policy, up, b0, s0);
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        violations[ep] = sum(hist.reward_hist .< 0.) #+ Int(n_steps(hist) >= max_steps)
    end
    return rewards, steps, violations
end

include("mdp_models/car_mdp/high_fidelity.jl")

println("EVALUATING IN HIGH FIDELITY ENVIRONMENT")

@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, rand_pol, n_ep=1000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)

# println("Using underlying MDP")

# umdp = UnderlyingMDP(pomdp)
# @time rewards_mask, steps_mask, violations_mask = evaluation_loop(umdp, rand_pol, n_ep=1000, max_steps=100, rng=rng);
# print_summary(rewards_mask, steps_mask, violations_mask)

#### Training using DQN

# include("masked_dqn.jl")

# max_steps = 500000
# eps_fraction = 0.5 
# eps_end = 0.01 
# solver = DeepQLearningSolver(max_steps = max_steps, eps_fraction = eps_fraction, eps_end = eps_end,
#                        lr = 0.0001,                    
#                        batch_size = 32,
#                        target_update_freq = 5000,
#                        max_episode_length = 200,
#                        train_start = 40000,
#                        buffer_size = 400000,
#                        eval_freq = 10000,
#                        arch = QNetworkArchitecture(conv=[], fc=[32,32,32]),
#                        double_q = true,
#                        dueling = true,
#                        prioritized_replay = true,
#                        exploration_policy = masked_linear_epsilon_greedy(max_steps, eps_fraction, eps_end, mask),
#                        evaluation_policy = masked_evaluation(mask),
#                        verbose = true,
#                        logdir = "carmdp-log/log",
#                        rng = rng)

# policy = solve(solver, mdp)