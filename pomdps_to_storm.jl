function write_storm_mdp{S,A}(mdp::MDP{S,A}, filename="mdp.tra")
    try
        run(`rm $filename`)
    catch
    end
    # create an ordered list of states for fast iteration
    states = ordered_states(mdp)
    open(filename, "w") do f
        write(f, "mdp \n")
        for s in states
            si = stateindex(mdp, s)
            si -= 1 # 0-indexed
            for a in actions(mdp)
                ai = actionindex(mdp, a) - 1
                d = transition(mdp, s, a)
                for sp in states
                    spi = stateindex(mdp, sp) - 1
                    prob = pdf(d, sp)
                    if prob != 0
                        line = string(si, " ", ai, " ", spi, " ", prob, "\n")
                        write(f, line)
                    end
                end
            end
        end
    end
end

function write_storm_mdp(pomdp::POMDP, filename="pomdp.tra")
    warn("Storm only supports MDPs, the transition file produced will be for the underlying MDP")
    try
        run(`rm $filename`)
    catch
    end
    # create an ordered list of states for fast iteration
    states = ordered_states(pomdp)
    open(filename, "w") do f
        write(f, "mdp \n")
        for s in states
            si = stateindex(pomdp, s)
            si -= 1 # 0-indexed
            for a in actions(pomdp)
                ai = actionindex(pomdp, a) - 1
                d = transition(pomdp, s, a)
                for sp in states
                    spi = stateindex(pomdp, sp) - 1
                    prob = pdf(d, sp)
                    if prob != 0
                        line = string(si, " ", ai, " ", spi, " ", prob, "\n")
                        write(f, line)
                    end
                end
            end
        end
    end
end

function write_storm_labels{S,A}(mdp::MDP{S, A}, labeling::Dict{S, String}, filename="mdp.lab")
    #TODO handle multiple labels for one state
    try
        run(`rm $filename`)
    catch
    end
    labels = Set{String}()
    for v in Set(values(labeling))
        labs = split(v, " ")
        for lab in labs
            push!(labels, lab)
        end
    end
    open(filename, "w") do f
        write(f, "#DECLARATION\n")
        for label in labels
            write(f, "$label ")
        end
        write(f, "\n")
        write(f, "#END\n")
        # states must be sorted by index
        # labeled_states = sort(collect(keys(labeling)), by=x->stateindex(mdp, x))
        for (i, s) in enumerate(ordered_states(mdp))
            if haskey(labeling, s)
                si = stateindex(mdp, s) - 1 #0 indexed
                write(f, string(si ," ", labeling[s], "\n"))
            end
        end
    end
end

# get the state action version of P
function get_proba(mdp::MDP, result)
    P = zeros(n_states(mdp))
    for (i, val) in enumerate(result[:get_values]())
        P[i] = val
    end
    return P
end

#XXX Not sure is this algorithm is mathematically sound!!! (should technically be over the product MDP)
function get_state_action_proba(mdp::MDP, P::Vector{Float64}, threshold::Float64)
    P_map = zeros(n_states(mdp), n_actions(mdp))
    states = ordered_states(mdp)
    actions = ordered_actions(mdp)
    for (si, s) in enumerate(states)
        P[si] == 0. ? continue : nothing             
        for (ai, a) in enumerate(actions)
            dist = transition(mdp, s, a)
            for (sp, p) in  weighted_iterator(dist)
                p == 0.0 ? continue : nothing # skip if zero prob
                spi = stateindex(mdp, sp)
                # P[spi] < threshold ? continue : nothing # skip if future state is risky
                P_map[si, ai] += p * P[spi]
            end
        end
#         if all(P_map[si, :] < threshold)
#             a = action(policy, s)
#             ai = actionindex(mdp, a)
#             P_map[si, ai] =
#         end
    end
    return P_map
end

# function get_state_action_proba(mdp::GridWorld, policy::Scheduler)
#     P_map = zeros(n_states(mdp), n_actions(mdp))
#     states = ordered_states(mdp)
#     for (si, s) in enumerate(states)
#         a = action(policy, s)
#         ai = actionindex(mdp, a)
#         P_map[si, ai] = 1.0
#     end
#     return P_map
# end

struct Scheduler{S, A} <: Policy
    mdp::MDP{S, A}
    _scheduler::PyObject
    scheduler::Vector{A}
    action_map::Vector{A}
end

function Scheduler{S, A}(mdp::MDP{S, A}, py_scheduler::PyObject)
    action_map = ordered_actions(mdp)
    scheduler = Vector{A}(n_states(mdp))
    for i=1:n_states(mdp)
        choice = py_scheduler[:get_choice](i-1)
        ai = choice[:get_deterministic_choice]() + 1
        scheduler[i] = action_map[ai]
    end
    return Scheduler(mdp, py_scheduler, scheduler, action_map)
end

function POMDPs.action{S, A}(policy::Scheduler{S, A}, s::S)
    si = stateindex(policy.mdp, s)
    return policy.scheduler[si]
    # choice = policy.scheduler[:get_choice](si-1)
    # if choice[:deterministic]
    #     ai = choice[:get_deterministic_choice]()
    # else
    #     error("SchedulerError: non deterministic choice")
    # end
    # return policy.action_map[ai+1]
end


struct SafetyMask{S, A}
    mdp::MDP{S, A}
    threshold::Float64
    scheduler::Scheduler
    risk_vec::Vector{Float64}
    risk_mat::Array{Float64, 2}
    actions::Vector{A}
end

function SafetyMask{S, A}(mdp::MDP{S, A}, scheduler::Scheduler, risk_vec::Vector{Float64}, threshold::Float64)
    risk_mat = get_state_action_proba(mdp, risk_vec, threshold)
    return SafetyMask(mdp, threshold, scheduler, risk_vec, risk_mat, ordered_actions(mdp))
end

# can be precomputed and stored when constructing the mask
function get_safe_actions{S,A}(mask::SafetyMask{S,A}, s::S)
    safe_actions = A[]
    sizehint!(safe_actions, n_actions(mask.mdp))
    si = stateindex(mask.mdp, s)
    safe = mask.risk_vec[si] > mask.threshold ? true : false
    if !safe # follow safe controller
        push!(safe_actions, action(mask.scheduler, s))
    else
        for (j, a) in enumerate(mask.actions)
            if mask.risk_mat[si, j] > mask.threshold
                push!(safe_actions, a)
            end
        end
    end
    return safe_actions
end

function get_safe_actions_binary{S, A}(mask::SafetyMask{S, A}, s::S)
    safe_actions = zeros(Bool, n_actions(mask.mdp))
    actions = ordered_actions(mask.mdp)
    si = stateindex(mdp, s)
    safe = mask.risk_vec[si] > mask.threshold ? true : false
    if !safe
        a = action(mask.scheduler, s)
        ai = actionindex(mask.mdp, a)
        safe_actions[ai] = true
    else
        for j in 1:n_actions(mask.mdp)
            if mask.risk_mat[si, j] > mask.threshold
                safe_actions[j] = true
            end
        end
    end
    return safe_actions
end


# Masked Eps Greedy Policy
struct MaskedEpsGreedyPolicy{S, A} <: Policy
    val::ValuePolicy # the greedy policy
    epsilon::Float64
    mask::SafetyMask{S, A}
    rng::AbstractRNG
end

MaskedEpsGreedyPolicy{S, A}(mdp::MDP{S, A}, epsilon::Float64, mask::SafetyMask{S, A}, rng::AbstractRNG) = MaskedEpsGreedyPolicy{S, A}(ValuePolicy(mdp), epsilon, mask, rng)

function POMDPs.action(policy::MaskedEpsGreedyPolicy, s)
    acts = get_safe_actions(policy.mask, s)
    if rand(rng) < policy.epsilon
        return rand(rng, acts)
    else
        acts[indmax(policy.val.value_table[stateindex(policy.val.mdp, s), actionindex(policy.val.mdp, a)] for a in acts)]
    end
end

struct MaskedValuePolicy <: Policy
    val::ValuePolicy
    mask::SafetyMask
end

function POMDPs.action(policy::MaskedValuePolicy, s)
    acts = get_safe_actions(policy.mask, s)
    return acts[indmax(policy.val.value_table[stateindex(policy.val.mdp, s), actionindex(policy.val.mdp, a)] for a in acts)]
end


### JUNK

# Compute Prmax[phi1 U phi2]
# bad_states = Set(mdp.reward_states[mdp.reward_values .< 0.])
# good_states = Set(mdp.reward_states[mdp.reward_values .> 0.])
# # good_states = Set(GridWorldState[])
# # good_states = Set([mdp.reward_states[indmax(mdp.reward_values)]])
# max_iterations = 100
# verbose = true
# function compute_reachability(mdp::GridWorld, bad_states::Set{GridWorldState}; max_iterations::Int64=100, verbose::Bool=false)
#     P_mat = zeros(n_states(mdp), n_actions(mdp))
#     P = zeros(n_states(mdp))
#     total_time = 0.0
#     iter_time = 0.0
#
#     # create an ordered list of states for fast iteration
#     states = ordered_states(mdp)
#
#     # main loop
#     for i = 1:max_iterations
#         residual = 0.0
#         tic()
#         # state loop
#         for (istate, s) in enumerate(states)
#             sub_aspace = actions(mdp, s)
#             if s ∈ bad_states
#                 P[istate] = 0.
#             elseif s ∈ good_states
#                 P[istate] = 1.
#             else
#                 old_P = P[istate] # for residual
#                 max_P = -Inf
#                 # action loop
#                 # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
#                 for a in iterator(sub_aspace)
#                     iaction = actionindex(mdp, a)
#                     dist = transition(mdp, s, a) # creates distribution over neighbors
#                     u = 0.0
#                     for (sp, p) in weighted_iterator(dist)
#                         p == 0.0 ? continue : nothing # skip if zero prob
#                         isp = stateindex(mdp, sp)
#                         u += p * P[isp]
#                     end
#                     new_P = u
#                     if new_P > max_P
#                         max_P = new_P
#                     end
#                     P_mat[istate, iaction] = new_P
#                 end # action
#                 # update the value array
#                 P[istate] = max_P
#                 diff = abs(max_P - old_P)
#                 diff > residual ? (residual = diff) : nothing
#             end
#         end # state
#         iter_time = toq()
#         total_time += iter_time
#         verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
#     end # main
#     return P, P_mat
# end
