function evaluation_loop(mdp::MDP, policy::Policy; n_ep::Int64 = 1000, max_steps::Int64 = 500, rng::AbstractRNG = Base.GLOBAL_RNG)
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    d0 = initial_state_distribution(mdp)
    @showprogress for ep=1:n_ep
        s0 = rand(rng, d0)
        hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        hist = simulate(hr, mdp, policy, s0)
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        violations[ep] = sum(hist.reward_hist .< 0.) #+ Int(n_steps(hist) >= max_steps)
    end
    return rewards, steps, violations
end

function evaluation_loop(pomdp::POMDP, policy::Policy; n_ep::Int64 = 1000, max_steps::Int64 = 500, rng::AbstractRNG = Base.GLOBAL_RNG)
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    up = PreviousObservationUpdater{obs_type(pomdp)}()
    @showprogress for ep=1:n_ep
        s0 = initial_state(pomdp, rng)
        o0 = generate_o(pomdp, s0, rng)
        b0 = initialize_belief(up, o0)
        hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        hist = simulate(hr, pomdp, policy, up, b0, s0);
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        violations[ep] = is_crash(hist.state_hist[end])#sum(hist.reward_hist .<= -1.) #+ Int(n_steps(hist) >= max_steps)
    end
    return rewards, steps, violations
end

function evaluate(pomdp::UrbanPOMDP, policy::Policy, max_steps::Int64, rng::AbstractRNG)
    s0 = initial_state(pomdp, rng)
    o0 = generate_o(pomdp, s0, rng)
    up = PreviousObservationUpdater{UrbanObs}()
    b0 = initialize_belief(up, o0)
    hr = HistoryRecorder(max_steps=max_steps, rng=rng)
    hist = simulate(hr, pomdp, policy, up, b0, s0)
    r = discounted_reward(hist)
    step = n_steps(hist)
    violation = is_crash(hist.state_hist[end])
    return r, step, violation
end

function parallel_evaluation(pomdp::POMDP, policy::Policy; n_ep::Int64 = 1000, max_steps::Int64 = 500, rng::AbstractRNG = Base.GLOBAL_RNG)
    rngs = Vector{AbstractRNG}(undef, n_ep)
    for i=1:n_ep
        rngs[i] = MersenneTwister(i)
    end
    res = pmap(x -> evaluate(pomdp, policy, max_steps, x), rngs)
    rewards, steps, violations = collect(zip(res...))
    return rewards, steps, violations
end

function print_summary(rewards, steps, violations)
    @printf("Summary for %d episodes: \n", length(rewards))
    @printf("Average reward: %.3f \n", mean(rewards))
    @printf("Average # of steps: %.3f \n", mean(steps))
    @printf("Average # of violations: %.3f \n", mean(violations)*100)
end
