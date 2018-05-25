#=
 Explicit MDP model of an urban environment with one pedestrian
(extension of the single crosswalk model)
=#

# State type
struct PedMDPState
    crash::Bool
    ego::VehicleState
    ped::VehicleState
end


# copy b to a
function Base.copy!(a::PedMDPState, b::PedMDPState)
    a.crash = b.crash
    a.ego = b.ego
    a.ped = b.ped
end

function Base.hash(s::PedMDPState, h::UInt64 = zero(UInt64))
    return hash(s.crash, hash(s.ego, hash(s.ped, h)))
end

function Base.:(==)(a::PedMDPState, b::PedMDPState)
    return a.crash == b.crash && a.ego == b.ego && a.ped == b.ped
end

# Action type
const PedMDPAction = UrbanAction


@with_kw mutable struct PedMDP <: MDP{PedMDPState, PedMDPAction}
    env::UrbanEnv = UrbanEnv(params=UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [10.0, 10., 10.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0))
    ego_type::VehicleDef = VehicleDef()
    ped_type::VehicleDef = VehicleDef(AgentClass.PEDESTRIAN, 1.0, 1.0)
    ped_model::DriverModel = ConstantPedestrian()
    max_acc::Float64 = 2.0
    pos_res::Float64 = 1.0
    vel_res::Float64 = 2.0
    vel_ped_res::Float64 = 1.0
    ego_start::Float64 = env.params.stop_line - ego_type.length/2
    ego_goal::LaneTag = LaneTag(2,1)
    off_grid::VecSE2 = VecSE2(UrbanEnv().params.x_min+VehicleDef().length/2, env.params.y_min+VehicleDef().width/2, 0)
    ΔT::Float64 = 0.5
    ped_birth::Float64 = 0.3
    collision_cost::Float64 = -1.0
    action_cost::Float64 = 0.
    goal_reward::Float64 = 1.
    γ::Float64 = 0.95
end

### REWARD MODEL ##################################################################################

function POMDPs.reward(mdp::PedMDP, s::PedMDPState, a::PedMDPAction, sp::PedMDPState)
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

function POMDPs.isterminal(mdp::PedMDP, s::PedMDPState)
    if s.crash
        return true
    elseif s.ego.posF.s >= get_end(mdp.env.roadway[mdp.ego_goal]) &&
       get_lane(mdp.env.roadway, s.ego).tag == mdp.ego_goal
       return true
   end
   return false
end

## Helpers

function POMDPs.discount(mdp::PedMDP)
    return mdp.γ
end

POMDPs.actions(mdp::PedMDP) = [PedMDPAction(-4.0), PedMDPAction(-2.0), PedMDPAction(0.0), PedMDPAction(2.0)]
POMDPs.n_actions(mdp::PedMDP) = 4

function POMDPs.action_index(mdp::PedMDP, action::PedMDPAction)
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

function get_off_the_grid(mdp::PedMDP)
    return VehicleState(mdp.off_grid, mdp.env.roadway, 0.)
end

# create labels for model checking
function labeling(mdp::PedMDP)
    labels = Dict{PedMDPState, Vector{String}}()
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

function POMDPs.convert_s(::Type{Vector{Float64}}, s::PedMDPState, mdp::PedMDP)
    n_features = 9 # x_ego, y_ego, th_ego, v_ego, x_ped, y_ped, th_ped, v_ped, crash
    z = zeros(n_features)
    z[1] = s.ego.posG.x / abs(mdp.env.params.x_min)
    z[2] = s.ego.posG.y / abs(mdp.env.params.y_min)
    z[3] = s.ego.posG.θ / π
    z[4] = s.ego.v / abs(mdp.env.params.speed_limit)
    z[5] = s.ped.posG.x / abs(mdp.env.params.x_min)
    z[6] = s.ped.posG.y / abs(mdp.env.params.y_min)
    z[7] = s.ped.posG.θ / π
    z[8] = s.ped.v / abs(mdp.env.params.speed_limit)
    z[9] = float(s.crash)
    return z
end