#=
 Explicit MDP model of an urban environment with one car
(extension of the single intersection model)
=#

# State type
struct CarMDPState
    crash::Bool
    ego::VehicleState
    car::VehicleState
    route::StaticVector
end

# copy b to a
function Base.copy!(a::CarMDPState, b::CarMDPState)
    a.crash = b.crash
    a.ego = b.ego
    a.car = b.car
    a.route = b.route
end

function Base.hash(s::CarMDPState, h::UInt64 = zero(UInt64))
    return hash(s.crash, hash(s.ego, hash(s.car, hash(s.route, h))))
end

function Base.:(==)(a::CarMDPState, b::CarMDPState)
    return a.crash == b.crash && a.ego == b.ego && a.car == b.car && a.route == b.route
end

# Action type
const CarMDPAction = UrbanAction


@with_kw mutable struct CarMDP <: MDP{CarMDPState, CarMDPAction}
    env::UrbanEnv = UrbanEnv(params=UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [10.0, 10., 10.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0))
    ego_type::VehicleDef = VehicleDef()
    car_type::VehicleDef = VehicleDef()
    car_model::DriverModel = RouteFollowingIDM(route=random_route(env, Base.GLOBAL_RNG), σ=1.0)
    max_acc::Float64 = 2.0
    pos_res::Float64 = 3.0
    vel_res::Float64 = 2.0
    ego_start::Float64 = env.params.stop_line - ego_type.length/2
    ego_goal::LaneTag = LaneTag(2,1)
    off_grid::VecSE2 = VecSE2(UrbanEnv().params.x_min+VehicleDef().length/2, env.params.y_min+VehicleDef().width/2, 0)
    ΔT::Float64 = 0.5
    car_birth::Float64 = 0.3
    collision_cost::Float64 = -1.0
    action_cost::Float64 = 0.
    goal_reward::Float64 = 1.
    γ::Float64 = 0.95
end

### REWARD MODEL ##################################################################################

function POMDPs.reward(mdp::CarMDP, s::CarMDPState, a::CarMDPAction, sp::CarMDPState)
    r = mdp.action_cost
    ego = sp.ego
    if sp.crash
        r += mdp.collision_cost
    end
    if ego.posF.s >= get_end(mdp.env.roadway[mdp.ego_goal]) &&
       get_lane(mdp.env.roadway, ego).tag == mdp.ego_goal
        r += mdp.goal_reward
    end
    return r
end

function POMDPs.isterminal(mdp::CarMDP, s::CarMDPState)
    if s.crash
        return true
    elseif s.ego.posF.s >= get_end(mdp.env.roadway[mdp.ego_goal]) &&
       get_lane(mdp.env.roadway, s.ego).tag == mdp.ego_goal
       return true
   end
   return false
end

## Helpers

function POMDPs.discount(mdp::CarMDP)
    return mdp.γ
end

POMDPs.actions(mdp::CarMDP) = [CarMDPAction(-4.0), CarMDPAction(-2.0), CarMDPAction(0.0), CarMDPAction(2.0)]
POMDPs.n_actions(mdp::CarMDP) = 4

function POMDPs.action_index(mdp::CarMDP, action::CarMDPAction)
    if action.acc == -4.0
        return 1
    elseif action.acc == -2.0
        return 2
    elseif action.acc == 0.
        return 3
    else
        return 4
    end
end

function get_off_the_grid(mdp::CarMDP)
    return VehicleState(mdp.off_grid, mdp.env.roadway, 0.)
end

function AutomotivePOMDPs.random_route(env::UrbanEnv, rng::AbstractRNG)
    start_lanes = get_start_lanes(env.roadway)
    ego_lanes = Set([env.roadway[tag] for tag in get_ego_route(env)])
    start_lanes = setdiff(start_lanes, ego_lanes)
    start_lane = rand(rng, start_lanes)
    return random_route(rng, env.roadway, start_lane)
end


# create labels for model checking
function labeling(mdp::CarMDP)
    labels = Dict{CarMDPState, Vector{String}}()
    for s in ordered_states(mdp)
        if s.crash
            labels[s] = ["crash"]
        elseif s.ego.posF.s >= get_end(mdp.env.roadway[mdp.ego_goal]) &&
            get_lane(mdp.env.roadway, s.ego).tag == mdp.ego_goal
            labels[s] = ["goal"]
        end
    end
    return labels
end