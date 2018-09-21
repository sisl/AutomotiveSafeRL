const Obs = Vector{Float64}
const Traj = Vector{Vector{Float64}}

function generate_trajectory(pomdp::UrbanPOMDP, policy::Policy, max_steps::Int64, rng::AbstractRNG)
    s0 = initial_state(pomdp, rng)
    o0 = generate_o(pomdp, s0, rng)
    up = FastPreviousObservationUpdater{UrbanObs}()
    b0 = initialize_belief(up, o0)
    hr = HistoryRecorder(max_steps=max_steps, rng=rng)
    hist = simulate(hr, pomdp, policy, up, b0, s0)
    # extract data from the history 
    X = Vector{SVector{length(o0), Float64}}(max_steps+1)
    fill!(X, zeros(length(o0)))
    X[1] = o0 
    X[2:n_steps(hist)+1] = hist.observation_hist
    Y = Vector{SVector{length(o0)-4, Float64}}(max_steps+1)
    fill!(Y, zeros(length(o0) - 4))
    svec = convert_s.(Vector{Float64}, hist.state_hist, pomdp) 
    Y[1:n_steps(hist)+1] = [s[5:end] for s in svec]
    return X, Y
end

function collect_set(pomdp::UrbanPOMDP, policy::Policy, max_steps::Int64, rng::AbstractRNG, n_set)
    X_batch = Vector{Traj}(n_set)
    Y_batch = Vector{Traj}(n_set)
    @showprogress for i=1:n_set
        X_batch[i], Y_batch[i] = generate_trajectory(pomdp, policy, max_steps, rng)
    end
    return X_batch, Y_batch
end



# # init continuous state mdp 
# mdp = PedCarMDP(pos_res=2.0, vel_res=2., ped_birth=0.7, car_birth=0.7)
# pomdp = UrbanPOMDP(env=mdp.env,
#                     sensor = GaussianSensor(false_positive_rate=0.05, 
#                                             pos_noise = LinearNoise(min_noise=0.5, increase_rate=0.05), 
#                                             vel_noise = LinearNoise(min_noise=0.5, increase_rate=0.05)),
#                    ego_goal = LaneTag(2, 1),
#                    max_cars=1, 
#                    max_peds=1, 
#                    car_birth=0.7, 
#                    ped_birth=0.7, 
#                    obstacles=false, # no fixed obstacles
#                    lidar=false,
#                    ego_start=20,
#                    ΔT=0.1)

# rng = MersenneTwister(1)
# policy = RandomPolicy(rng, pomdp, VoidUpdater())

# max_steps = 100
# n_train = 2000
# train_X, train_Y = collect_set(pomdp, policy, max_steps, rng, n_train)

# n_val = 1000
# val_X, val_Y = collect_set(pomdp, policy, max_steps, rng, n_val)
