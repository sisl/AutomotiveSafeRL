#=
Helpers to use masking during training with DQN 
Uses a global variable mask that should be defined in main prior to including this file 
=#


function best_action(acts::Vector{A}, val::Array{Float32, 2}, problem::M) where {A, M <: Union{POMDP, MDP}}
    all_actions = actions(problem)
    best_ai = 1 
    best_val = val[best_ai]
    for a in acts 
        ai = actionindex(problem, a)
        if val[ai] > best_val 
            best_val = val[ai]
            best_ai = ai 
        end
    end
    return all_actions[best_ai]::A
end


# define exploration policy with masking
function masked_linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float64, eps_end::Float64, mask::M) where M <: Union{SafetyMask, JointMask}
    # define function that will be called to select an action in DQN
    # only supports MDP environments
    function action_masked_epsilon_greedy(policy::AbstractNNPolicy, env::POMDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
        eps = update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        acts = safe_actions(pomdp, mask, obs)
        val = get_value!(policy, obs)
        if rand(rng) < eps
            return (rand(rng, acts), eps)
        else
            return (best_action(acts, val, env.problem), eps)
        end
    end
    return action_masked_epsilon_greedy
end

function masked_evaluation(mask::M) where M <: Union{SafetyMask, JointMask}
    function masked_evaluation_policy(policy::AbstractNNPolicy, env::POMDPEnvironment, n_eval::Int64, max_episode_length::Int64, verbose::Bool)
        avg_r = 0 
        for i=1:n_eval
            done = false 
            r_tot = 0.0
            step = 0
            obs = reset(env)
            DeepQLearning.reset_hidden_state!(policy)
            while !done && step <= max_episode_length
                acts = safe_actions(pomdp, mask, obs)
                val = get_value!(policy, obs)
                act = best_action(acts, val, env.problem)
                obs, rew, done, info = step!(env, act)
                r_tot += rew 
                step += 1
            end
            avg_r += r_tot 
        end
        if verbose
            println("Evaluation ... Avg Reward ", avg_r/n_eval)
        end
        avg_r /= n_eval
        return  avg_r
    end
    return masked_evaluation_policy
end


###

struct MaskedDQNPolicy{P <: POMDP, M <: Union{SafetyMask, JointMask}} <: Policy
    problem::P
    q::DQNPolicy
    mask::M
end

function POMDPs.action(policy::MaskedDQNPolicy, s)
    acts = safe_actions(policy.problem, policy.mask, s)
    val = value(policy.q, s)
    act = best_action(acts, val, policy.problem)
    return act 
end
