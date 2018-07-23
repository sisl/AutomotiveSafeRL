# XXX uses global variable POMDP
# function MDPModelChecking.safe_actions(mask::SafetyMask{CarMDP, CarMDPAction}, o::UrbanObs, ped_id=2)
#     s = obs_to_scene(pomdp, o)
#     return safe_actions(pomdp, mask, s, ped_id)
# end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{CarMDP, P}, o::UrbanObs) where P <: Policy
    s = obs_to_scene(pomdp, o)
    return safe_actions(pomdp, mask, s, CAR_ID)
end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{CarMDP, P}, o::Array{Float64, 2}) where P <: Policy
    d, dd = size(o)
    @assert dd == 1
    return safe_actions(mask, o[:], CAR_ID)
end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{CarMDP, P}, s::UrbanState, car_id::Int64) where P <: Policy
    s_mdp = get_mdp_state(mask.mdp, pomdp, s, car_id)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    # compute risk vector
    # si = state_index(mdp, itp_states[indmax(itp_weights)])
    # p_sa = mask.risk_mat[si, :]
#     p_sa_itp = zeros(length(itp_states), n_actions(mask.mdp))
#     for (i, ss) in enumerate(itp_states)
#         si = state_index(mask.mdp, ss)
#         p_sa_itp[i, :] += itp_weights[i]*mask.risk_mat[si,:]
#     end
#     p_sa = minimum(p_sa_itp, 1)
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        vals = value_vector(mask.policy, ss)
        p_sa += itp_weights[i]*vals
    end
    safe_acts = UrbanAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    action_space = actions(mask.mdp)
    if maximum(p_sa) <= mask.threshold
        push!(safe_acts, action_space[indmax(p_sa)])
    else
        for (j, a) in enumerate(action_space)
            if p_sa[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    # println("coucou ")
    # global debug_i
    # println("Safe acts $([a.acc for a in safe_acts])")
    # println(" i ", debug_i)
    # debug_i += 1
    return safe_acts
end


#XXX uses global variable POMDP
# function MDPModelChecking.safe_actions(mask::SafetyMask{PedMDP, PedMDPAction}, o::UrbanObs, ped_id=101)
#     s = obs_to_scene(pomdp, o)
#     return safe_actions(mask, s, ped_id)
# end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedMDP, P}, o::UrbanObs) where P <: Policy
    s = obs_to_scene(pomdp, o)
    return safe_actions(mask, s, PED_ID)
end


function MDPModelChecking.safe_actions(mask::SafetyMask{PedMDP, P}, o::Array{Float64, 2}) where P <: Policy
    d, dd = size(o)
    @assert dd == 1
    return safe_actions(mask, o[:])
end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedMDP, P},s::UrbanState) where P <: Policy   
    return safe_actions(mask, s, PED_ID)
end

function MDPModelChecking.safe_actions(mask::SafetyMask{PedMDP, P}, s::UrbanState, ped_id) where P <: Policy
    s_mdp = get_mdp_state(mask.mdp, s, ped_id)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    # compute risk vector
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        vals = value_vector(mask.policy, ss)
        p_sa += itp_weights[i]*vals
    end
    safe_acts = UrbanAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    action_space = actions(mask.mdp)
    if maximum(p_sa) <= mask.threshold
        push!(safe_acts, action_space[indmax(p_sa)])
    else
        for (j, a) in enumerate(action_space)
            if p_sa[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    return safe_acts
end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, o::UrbanObs) where P <: Policy
    s = obs_to_scene(pomdp, o)
    return safe_actions(pomdp, mask, s, PED_ID, CAR_ID)
end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, o::Array{Float64, 2}) where P <: Policy
    d, dd = size(o)
    @assert dd == 1
    return safe_actions(mask, o[:], PED_ID, CAR_ID)
end

# function MDPModelChecking.value_vector(policy::ValueIterationPolicy, s::PedCarMDPState)
#     if !s.crash && isterminal(policy.mdp, s)
#         return ones(n_actions(policy.mdp))
#     else
#         si = state_index(policy.mdp, s)
#         return policy.qmat[si, :]
#     end
# end

# function MDPModelChecking.value_vector(policy::LocalApproximationValueIterationPolicy, s::PedCarMDPState)
#     if !s.crash && isterminal(policy.mdp, s)
#         return ones(n_actions(policy.mdp))
#     else
#         q = zeros(n_actions(policy.mdp))
#         for i = 1:n_actions(policy.mdp)
#             q[i] = action_value(policy, s, policy.action_map[i])
#         end
#         return q
#     end
# end

function MDPModelChecking.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, s::UrbanState, ped_id, car_id) where P <: Policy
    s_mdp = get_mdp_state(mask.mdp, pomdp, s, ped_id, car_id)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    action_space = actions(mask.mdp)
    # compute risk vector
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        vals = value_vector(mask.policy, ss)
        p_sa += itp_weights[i]*vals
    end
    # println(p_sa)
    safe_acts = UrbanAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    if maximum(p_sa) <= mask.threshold
        push!(safe_acts, action_space[indmax(p_sa)])
    else
        for (j, a) in enumerate(action_space)
            if p_sa[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    return safe_acts
end

function POMDPToolbox.action_info(policy::MaskedEpsGreedyPolicy{M}, s) where M <: SafetyMask
    return action(policy, s), [safe_actions(policy.mask, s), s]
end

# ## new policy type to work with UrbanPOMDP

struct RandomMaskedPOMDPPolicy{M} <: Policy 
    mask::M
    pomdp::UrbanPOMDP
    rng::AbstractRNG
end

function POMDPs.action(policy::RandomMaskedPOMDPPolicy, s)
    acts = safe_actions(policy.pomdp, policy.mask, s)
    if isempty(acts)
        def_a = UrbanAction(-4.0)
        # warn("WARNING: No feasible action at this step, choosing default action $(def_a.acc)m/s^2")
        return def_a
    end
    return rand(policy.rng, acts)
end

function POMDPToolbox.action_info{M}(policy::RandomMaskedPOMDPPolicy{M}, s)
    return action(policy, s), safe_actions(policy.pomdp, policy.mask, s)
end


struct JointMask{P <: MDP, M <: SafetyMask, I}
    problems::Vector{P}
    masks::Vector{M}
    ids::Vector{I}
end

function MDPModelChecking.safe_actions{S}(pomdp::UrbanPOMDP, mask::JointMask, s::S)
    acts = intersect([safe_actions(pomdp, m, s) for m in mask.masks]...) 
    if isempty(acts)
        return UrbanAction[UrbanAction(-4.0)]
    end
    return acts       
end

# struct MaskedEpsGreedyPolicyPOMDP{M} <: Policy where {M <: SafetyMask}
#     val::ValuePolicy # the greedy policy
#     epsilon::Float64
#     mask::M
#     pomdp::UrbanPOMDP
# end

# MaskedEpsGreedyPolicyPOMDP{S, A, M}(mdp::MDP{S, A}, pomdp::UrbanPOMDP, epsilon::Float64, mask::M, rng::AbstractRNG) = MaskedEpsGreedyPolicy(ValuePolicy(mdp), epsilon, mask, rng)

# function POMDPs.action(policy::MaskedEpsGreedyPolicyPOMDP, s)
#     acts = safe_actions(policy.pomdp, policy.mask, s)
#     if rand(policy.rng) < policy.epsilon
#         return rand(policy.rng, acts)
#     else
#         return best_action(acts, policy.val, s)
#     end
# end
