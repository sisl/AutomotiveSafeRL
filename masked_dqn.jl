#=
Helpers to use masking during training with DQN 
Uses a global variable mask that should be defined in main prior to including this file 
=#


function best_action(acts::Vector{A}, val::Array{Float32, 2}, mdp::M) where {A, M <: MDP}
    all_actions = actions(mdp)
    best_ai = 1 
    best_val = val[best_ai]
    for a in acts 
        ai = action_index(mdp, a)
        if val[ai] > best_val 
            best_val = val[ai]
            best_ai = ai 
        end
    end
    return all_actions[best_ai]
end


# define exploration policy with masking
function masked_linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float64, eps_end::Float64, mask::SafetyMask)
    # define function that will be called to select an action in DQN
    # only supports MDP environments
    function action_masked_epsilon_greedy(policy::DQNPolicy, env::MDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
        eps = update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        s = env.state
        acts = safe_actions(mask, s)
        val = get_value(policy, obs)
        if rand(rng) < eps
            return (rand(rng, acts), eps)
        else
            return (best_action(acts, val, env.problem), eps)
        end
    end
    return action_masked_epsilon_greedy
end

function masked_evaluation(mask::SafetyMask)
    function masked_evaluation_policy(policy::DQNPolicy, env::MDPEnvironment, n_eval::Int64, max_episode_length::Int64, verbose::Bool)
        avg_r = 0 
        for i=1:n_eval
            done = false 
            r_tot = 0.0
            step = 0
            obs = reset(env)
            while !done && step <= max_episode_length
                s = env.state
                acts = safe_actions(mask, s)
                val = get_value(policy, obs)
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
        return  avg_r /= n_eval
    end
    return masked_evaluation_policy
end
