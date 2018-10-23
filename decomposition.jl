# utility decomposition: 

struct DecMaskedPolicy{A <: Policy, M <: SafetyMask, P <: Union{MDP, POMDP}} <: Policy
    policy::A
    mask::M
    problem::P
    op # reduction operator
end

function POMDPs.value(policy::DecMaskedPolicy, dec_belief::Dict)  # no hidden state!
    return reduce(policy.op, action_value(policy.policy, b) for (_,b) in dec_belief)
end

function POMDPs.action(p::DecMaskedPolicy, b::Dict)
    safe_acts = safe_actions(p.problem, p.mask, b)
    val = value(p, b)
    act = best_action(safe_acts, val, policy.problem)
    return act
end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, b::Dict{I, PedCarRNNBelief}) where {P <: Policy,I}
    reduce(intersect, safe_actions(pomdp, mask, bel, ids[2], ids[1]) for (ids, bel) in b)
end

function POMDPModelTools.action_info(p::DecMaskedPolicy, b::Dict)
    # println("Beliefs keys: ", keys(b))
    safe_acts = safe_actions(p.problem, p.mask, b)
    # compute probas
    probs = sum(compute_probas(p.problem, p.mask, bel, ids[2], ids[1]) for (ids, bel) in b)./length(b)
    val = value(p, b)
    act = best_action(safe_acts, val, policy.problem)
    return act, (safe_acts, probs)
end

# belief decomposition 

struct DecUpdater{P <: POMDP, I, U<:Updater} <: Updater
    problem::P
    updaters::Dict{I, U}
end

function POMDPs.update(up::DecUpdater, bold::Dict{NTuple{3, Int64}, PedCarRNNBelief}, a::UrbanAction, o::UrbanObs)
    ego, car_map, ped_map, obs_map = split_o(o, up.problem)
    augment_with_absent_state!(up.problem, car_map, ego, max(2, length(car_map)+1))
    augment_with_absent_state!(up.problem, ped_map, ego, 101 + length(ped_map))
    dec_o = create_pedcar_states(ego, car_map, ped_map, obs_map)
    # println("decomposed o: ", keys(dec_o))
    ref_up = up.updaters[(0,0,0)]
    bnew = Dict{NTuple{3, Int64}, PedCarRNNBelief}()
    for (obs_id, obs) in dec_o
        if haskey(bold, obs_id) 
            @assert haskey(up.updaters, obs_id) "KeyError: $obs_id keys in old belief: $(keys(bold)), keys in updater: $(keys(up.updaters))"# should have an associated filter 
            bnew[obs_id] = update(up.updaters[obs_id], bold[obs_id], a, obs)
        else # instantiate new filter 
            up.updaters[obs_id] = PedCarRNNUpdater(deepcopy(ref_up.models), ref_up.mdp, ref_up.pomdp) # could do something smarter than deepcopy
            reset_updater!(up.updaters[obs_id])
            init_belief = PedCarRNNBelief(Vector{Vector{Float64}}(undef, n_models), obs)
            bnew[obs_id] = update(up.updaters[obs_id], init_belief, a, obs)
        end
    end
    return bnew
end   

function augment_with_absent_state!(pomdp::UrbanPOMDP, dict::OrderedDict{Int64, Vector{Float64}}, ego::Vector{Float64}, id::Int64)
    ego_x, ego_y, theta, v = ego
    pos_off =  get_off_the_grid(pomdp)
    max_ego_dist = get_end(pomdp.env.roadway[pomdp.ego_goal])
    dict[id] = [pos_off.posG.x/max_ego_dist - ego_x,
                    pos_off.posG.y/max_ego_dist - ego_y,
                    pos_off.posG.θ/float(pi),
                    0. ]
    return dict 
end

function create_pedcar_states(ego, car_map, ped_map, obs_map)
    decomposed_state = Dict{NTuple{3, Int64}, Vector{Float64}}()
    for (car_id, car) in car_map
        for (ped_id, ped) in ped_map
            for (obs_id, obs) in obs_map
                decomposed_state[(car_id, ped_id, obs_id)] = vcat(ego, car, ped, obs)
            end
        end
    end
    return decomposed_state
end


# utility decomposition 

# struct DecomposedMask{CM <: SafetyMask, PM <: SafetyMask}
#     pomdp::UrbanPOMDP
#     car_mask::CM
#     ped_mask::PM
# end

# function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::DecomposedMask, o::UrbanObs)
#     s = obs_to_scene(pomdp, o)
#     action_sets = Vector{Vector{UrbanAction}}()
#     np = 0
#     nc = 0
#     current_ids = keys(pomdp.models)
#     for veh in s 
#         if veh.id == EGO_ID
#             continue
#         elseif veh.def.class == AgentClass.PEDESTRIAN
#             safe_acts = safe_actions(mask.ped_mask, s, veh.id)
#             push!(action_sets, safe_acts)
#             np += 1
#             # println("For veh $(veh.id) at $(veh.state.posF), safe actions $safe_acts")
#         elseif veh.def.class == AgentClass.CAR
#             # trick
#             veh_ = veh.id
#             if !(haskey(pomdp.models, veh.id))
#                 veh_ = veh.id == 2 ? 3 : 2
#             end
#             safe_acts = safe_actions(pomdp, mask.car_mask, s, veh_)
#             nc += 1
#             push!(action_sets, safe_acts)
#             # println("For veh $(veh.id) at $(veh.state.posF), $(veh.state.v), safe actions $safe_acts")
#         end
#     end
#     # add absent pedestrian and absent car
#     # if np < pomdp.max_peds
#     #     push!(action_sets, safe_actions(mask.ped_mask, s, 102+pomdp.max_peds))
#     # end
#     # if nc < pomdp.max_cars
#     #     push!(action_sets, safe_actions(pomdp, mask.car_mask, s, 3 + pomdp.max_cars))
#     # end
#     if isempty(action_sets)
#         return actions(pomdp)
#     end
#     # take intersection
#     acts = intersect(action_sets...)
#     if isempty(acts)
#         return UrbanAction[UrbanAction(-4.0)]
#     end
#     return acts 
# end


### Template for decomposition method 

# struct DecPolicy{P <: Policy, M <: Union{MDP, POMDP}, A} <: Policy
#     policy::P # the single agent policy
#     problem::M # the pomdp definition
#     action_map::Vector{A}
#     op # the reduction operator for utiliy fusion (e.g. sum or min)
# end


# function action_values(policy::DecPolicy, dec_belief::Dict)  # no hidden state!
#     return reduce(policy.op, action_values(policy.policy, b) for (_,b) in dec_belief)
# end

# function POMDPs.action(p::DecPolicy, b::Dict)
#     vals = action_values(p, b)
#     ai = indmax(vals)
#     return p.action_map[ai]
# end

# function action_values(p::AlphaVectorPolicy, b::SparseCat)
#     num_vectors = length(p.alphas)
#     utilities = zeros(n_actions(p.pomdp), num_vectors)
#     action_counts = zeros(n_actions(p.pomdp))
#     for i = 1:num_vectors
#         ai = actionindex(p.pomdp, p.action_map[i])
#         action_counts[ai] += 1
#         utilities[ai, i] += sparse_cat_dot(p.pomdp, p.alphas[i], b)
#     end
#     utilities ./= action_counts
#     return maximum(utilities, dims=2)
# end

# # perform dot product between an alpha vector and a sparse cat object
# function sparse_cat_dot(problem::POMDP, alpha::Vector{Float64}, b::SparseCat)
#     val = 0.
#     for (s, p) in weighted_iterator(b)
#         si = stateindex(problem, s)
#         val += alpha[si]*p
#     end
#     return val 
# end