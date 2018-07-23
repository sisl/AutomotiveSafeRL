using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DiscreteValueIteration
using ProgressMeter, Parameters, JLD
using LocalFunctionApproximation, LocalApproximationValueIteration
using NearestNeighbors, StatsBase

function sample_points(mdp::PedCarMDP, n_samples::Int64, rng::AbstractRNG)
    # sample points for the approximator
    state_space = states(mdp)
    n_routes = 4
    n_features = 4
    nd = n_features*3 + n_routes + 1 
    sampled_states = sample(rng, 1:length(state_space), n_samples, replace=false)
    points = Vector{SVector{nd, Float64}}(n_samples)
    for (i, si) in enumerate(sampled_states)
        z = convert_s(Vector{Float64}, state_space[si], mdp)
        points[i] = SVector{nd, Float64}(z...)
    end
    return points, sampled_states
end

function convert_states(mdp::PedCarMDP, sampled_states::Vector{Int64})
    n_routes = 4
    n_features = 4
    nd = n_features*3 + n_routes + 1
    state_space = states(mdp)
    points = Vector{SVector{nd, Float64}}(length(sampled_states))
    for (i, si) in enumerate(sampled_states)
        z = convert_s(Vector{Float64}, state_space[si], mdp)
        points[i] = SVector{nd, Float64}(z...)
    end
    return points
end
rng = MersenneTwister(1)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(env = env, pos_res=2., vel_res=2., ped_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))

# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

N_SAMPLES = 200000
k = 6
knnfa = nothing
sampled_states = nothing
policy_file = "pc_lavi_fine.jld"
if isfile(policy_file)
    data = load(policy_file)
    sampled_states = data["sampled_states"]
    points = convert_states(mdp, sampled_states)
    nntree = KDTree(points)
    knnfa = LocalNNFunctionApproximator(nntree, points, k)
    set_all_interpolating_values(knnfa, data["values"])
else
    points, sampled_states = sample_points(mdp, N_SAMPLES, rng)
    nntree = KDTree(points)
    knnfa = LocalNNFunctionApproximator(nntree, points, k)
end

approx_solver = LocalApproximationValueIterationSolver(knnfa, verbose=true, max_iterations=40, is_mdp_generative=false)
policy = solve(approx_solver, mdp)

JLD.save(policy_file, "sampled_states", sampled_states, "values", policy.interp.nnvalues ) 
