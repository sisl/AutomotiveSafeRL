const Obs = Vector{Float64}
const Traj = Vector{Vector{Float64}}

function generate_trajectory(pomdp::UrbanPOMDP, policy::Policy, max_steps::Int64, rng::AbstractRNG)
    n_features = 5 # absence/presence
    n_obstacles = pomdp.max_obstacles 
    max_ego_dist = get_end(pomdp.env.roadway[pomdp.ego_goal])
    speed_limit = pomdp.env.params.speed_limit
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
    # build labels 
    label_dims = (n_features)*(pomdp.max_cars + pomdp.max_peds)
    Y = [zeros(label_dims) for i=1:(max_steps+1)]
    # svec = convert_s.(Vector{Float64}, hist.state_hist, pomdp) 
    for (i, s) in enumerate(hist.state_hist)
        ego = get_ego(s).state
        sorted_vehicles = sort!([veh for veh in s], by=x->x.id)
        for veh in sorted_vehicles
            if veh.id == EGO_ID
                continue
            end
            if veh.def.class == AgentClass.CAR
                @assert veh.id <= pomdp.max_cars+1 "$(veh.id)"
                Y[i][n_features*(veh.id-1) - 4] = (veh.state.posG.x - ego.posG.x)/max_ego_dist
                Y[i][n_features*(veh.id-1) - 3] = (veh.state.posG.y - ego.posG.y)/max_ego_dist
                Y[i][n_features*(veh.id-1) - 2] = veh.state.posG.θ/float(pi)
                Y[i][n_features*(veh.id-1) - 1] = veh.state.v/speed_limit
                Y[i][n_features*(veh.id-1)] = 1.0
            end
            if veh.def.class == AgentClass.PEDESTRIAN
                Y[i][n_features*(veh.id - 100 + pomdp.max_cars) - 4] = (veh.state.posG.x - ego.posG.x)/max_ego_dist
                Y[i][n_features*(veh.id - 100 + pomdp.max_cars) - 3] = (veh.state.posG.y - ego.posG.y)/max_ego_dist
                Y[i][n_features*(veh.id - 100 + pomdp.max_cars) - 2] = veh.state.posG.θ/float(pi)
                Y[i][n_features*(veh.id - 100 + pomdp.max_cars) - 1] = veh.state.v/speed_limit
                Y[i][n_features*(veh.id - 100 + pomdp.max_cars)] = 1.0
            end
        end    
    end
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

mutable struct RandomHoldPolicy{P<:POMDP, A} <: Policy 
    pomdp::P
    hold_time::Int64
    count::Int64
    current_action::A
    rng::AbstractRNG
end

function POMDPs.action(pol::RandomHoldPolicy, s)
    if pol.count % pol.hold_time == 0
        pol.current_action = rand(rng, actions(pol.pomdp))
        pol.count = 1
    else
        pol.count += 1
    end
    return pol.current_action 
end

function process_prediction(pomdp::UrbanPOMDP, b::Vector{Float64}, o::Vector{Float64})
    n_features = 4
    n_obstacles = pomdp.max_obstacles
    b_ = zeros(length(o)) # should be 12 + obstacles
    b_[1:4] = o[1:4] # ego state fully observable
    # get car state from b
    car_presence = b[5]
    if 1/(1 + exp(-car_presence)) > 0.5 # softmax 
        b_[n_features+1:2*n_features] = b[1:4]
    else
        # absent
        b_[n_features+1:2*n_features] = normalized_off_the_grid_pos(pomdp, o[1], o[2])
    end
    
    # get ped state from b 
    ped_presence = b[10]
    if 1/(1 + exp(-ped_presence)) > 0.5 # softmax 
        b_[2*n_features+1:3*n_features] = b[6:9]
    else
        # absent 
        b_[2*n_features+1:3*n_features] = normalized_off_the_grid_pos(pomdp, o[1], o[2])
    end
            
    b_[end - n_features*n_obstacles + 1:end] = o[end - n_features*n_obstacles + 1:end]
    return b_
end

function normalized_off_the_grid_pos(pomdp::UrbanPOMDP,normalized_ego_x::Float64, normalized_ego_y::Float64)
    pos_off = get_off_the_grid(pomdp)
    max_ego_dist = get_end(pomdp.env.roadway[pomdp.ego_goal])
    return [pos_off.posG.x/max_ego_dist - normalized_ego_x, pos_off.posG.y/max_ego_dist - normalized_ego_y, pos_off.posG.θ, 0.]
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
