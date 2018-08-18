N_PROCS=56
addprocs(N_PROCS)
rng = MersenneTwister(1)
@everywhere begin 
    using POMDPs, POMDPToolbox, DiscreteValueIteration
    using AutomotivePOMDPs, AutomotiveDrivingModels
    using JLD, StaticArrays
    using PedCar
    using MDPModelChecking
    using LocalApproximationValueIteration
    using ProgressMeter
    function DiscreteValueIteration.ind2state(mdp::PedCar.PedCarMDP, si::Int64)
        PedCar.ind2state(mdp, si)
    end

end # @everywhere

include("masking.jl")
include("util.jl")


@everywhere params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
@everywhere env = UrbanEnv(params=params);

@everywhere mdp = PedCar.PedCarMDP(env=env, pos_res=2.0, vel_res=2.0, ped_birth=0.7, car_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))
@everywhere init_transition!(mdp)

# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

solver = ParallelValueIterationSolver(n_procs=N_PROCS, max_iterations=10, belres=1e-4, include_Q=true, verbose=true)
policy_file = "pc_util_inter.jld"
if isfile(policy_file)
  data = load(policy_file)
  solver.init_util = data["util"]
end
policy = solve(solver, mdp)
JLD.save(policy_file, "util", policy.util, "qmat", policy.qmat, "pol", policy.policy)

threshold = 0.9999
mdp.collision_cost = -1.0
mask = SafetyMask(mdp, policy, threshold);
rand_pol = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng);
println("Evaluation in discretized environment: \n ")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, rand_pol, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)
mdp.collision_cost = 0.

solver.init_util = policy.util
policy = solve(solver, mdp) # resume


threshold = 0.9999
mdp.collision_cost = -1.0
mask = SafetyMask(mdp, policy, threshold);
rand_pol = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng);
println("Evaluation in discretized environment: \n ")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, rand_pol, n_ep=10000, max_steps=100, rng=rng);print_summary(rewards_mask, steps_mask, violations_mask)
mdp.collision_cost = 0.0

JLD.save(policy_file, "util", policy.util, "qmat", policy.qmat, "pol", policy.policy)
JLD.save("pedcar_policy3.jld", "policy", policy)
