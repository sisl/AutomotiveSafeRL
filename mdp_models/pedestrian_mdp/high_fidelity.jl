function interpolate_state(mdp::PedMDP, s::PedMDPState)
    # interpolate s in the grid
    ped_v = get_ped_vspace(mdp.env, mdp.vel_ped_res) 
    itp_ped, itp_ped_w = interpolate_pedestrian(s.ped, ped_v)
    ego_v = get_car_vspace(mdp.env, mdp.vel_res)
    itp_ego, itp_ego_w = interpolate_state(s.ego, ego_v)
    itp_states = Vector{PedMDPState}(length(itp_ego)*length(itp_ped))
    itp_w = Vector{Float64}(length(itp_states))
    k = 1
    for (i, p) in enumerate(itp_ped)
        for (j, e) in enumerate(itp_ego)
            crash = is_colliding(Vehicle(p, mdp.ped_type, 0), Vehicle(e, mdp.ego_type, 1))
            itp_states[k] = PedMDPState(crash, e, p)
            itp_w[k] = itp_ped_w[i]*itp_ego_w[j]
            k += 1
        end
    end
    @assert sum(itp_w) ≈ 1.
    return itp_states, itp_w
end

function car_roadway(env::UrbanEnv)
    params = env.params
    intersection_params = TInterParams(params.x_min, params.x_max, params.y_min, params.inter_x,
                                    params.inter_y, params.inter_width, params.inter_height,
                                    params. lane_width, params.nlanes_main, params.nlanes,
                                    params. stop_line, params.speed_limit, params.car_rate)
    car_roadway = gen_T_roadway(intersection_params)
    return car_roadway
end

function get_pedmdp_state(mdp::PedMDP, s::Scene, ped_id::Int64)
    ped_i = findfirst(s, ped_id)
    ped = Vehicle(get_off_the_grid(mask.mdp), mdp.ped_type, ped_id)
    if ped_i != 0
        ped = s[ped_i]
    end
    ego = get_ego(s)
    e_state = VehicleState(ego.state.posG, car_roadway(mdp.env), ego.state.v)
    p_lane = get_lane(mdp.env.roadway, ped.state)
    if p_lane.tag ∈ [LaneTag(17,1), LaneTag(18,1), LaneTag(19,1)]
        if ped.state.posG.θ == 0. ||ped.state.posG.θ == float(pi) # solve ambiguity in projection
            p_lane = mdp.env.ped_roadway[LaneTag(19,1)]
        end
        p_state = VehicleState(ped.state.posG, p_lane, mdp.env.ped_roadway, ped.state.v)
    else
        p_state = VehicleState(ped.state.posG, mdp.env.ped_roadway, ped.state.v)
    end
    return PedMDPState(is_colliding(ego, ped), e_state, p_state)
end

function render_interpolated_states(mdp::PedMDP, scene::Scene, ped_id=101)
    sp = deepcopy(scene)
    s = get_pedmdp_state(mdp, sp, ped_id)
    ped_v = get_ped_vspace(mdp.env, mdp.vel_ped_res) 
    itp_ped, itp_ped_w = interpolate_pedestrian(s.ped, ped_v)
    ego_v = get_car_vspace(mdp.env, mdp.vel_res)
    itp_ego, itp_ego_w = interpolate_state(s.ego, ego_v)
    colors = Dict{Int64, Colorant}(1=>COLOR_CAR_EGO, ped_id=>COLOR_CAR_OTHER)
    pid = ped_id+1
    for pi in itp_ped
        if pi.posG != mdp.off_grid
            push!(sp, Vehicle(pi, mdp.ped_type, pid))
            colors[pid] = MONOKAI["color2"]
            pid += 1            
        end
    end
    eid = EGO_ID+1
    for ei in itp_ego
        push!(sp, Vehicle(ei, mdp.ego_type, eid))
        colors[eid] = MONOKAI["color2"]
        eid += 1        
    end
    return AutoViz.render(sp, env, cam=FitToContentCamera(0.), car_colors = colors)    
end

mutable struct InterpolationOverlay <: SceneOverlay
    verbosity::Int
    color::Colorant
    font_size::Int
    id::Int
    mdp::PedMDP
    pomdp::UrbanPOMDP

    function InterpolationOverlay(mdp::PedMDP, pomdp::UrbanPOMDP, id::Int=101;
        verbosity::Int=1,
        color::Colorant=colorant"white",
        font_size::Int=20,
        )

        new(verbosity, color, font_size, id, mdp, pomdp)
    end
end

function AutoViz.render!(rendermodel::RenderModel, overlay::InterpolationOverlay, scene::Scene, env::OccludedEnv)
    transparency = 0.5
    s = get_pedmdp_state(overlay.mdp, scene, overlay.id)
    ped_v = get_ped_vspace(mdp.env, mdp.vel_ped_res) 
    itp_ped, itp_ped_w = interpolate_pedestrian(s.ped, ped_v)
    ego_v = get_car_vspace(mdp.env, mdp.vel_res)
    itp_ego, itp_ego_w = interpolate_state(s.ego, ego_v)
    for pi in itp_ped
        x, y, θ, v = pi.posG.x, pi.posG.y, pi.posG.θ, pi.v 
        color = RGBA(75./255, 66./255, 244./255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ped_type.length, overlay.mdp.ped_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end
    for ei in itp_ego
         x, y, θ, v = ei.posG.x, ei.posG.y, ei.posG.θ, ei.v 
        color = RGBA(75./255, 66./255, 244./255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ego_type.length, overlay.mdp.ego_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end    
end

# XXX uses global variable POMDP
function POMDPStorm.safe_actions(mask::SafetyMask{PedMDP, PedMDPAction}, o::UrbanObs, ped_id=101)
    s = obs_to_scene(pomdp, o)
    return safe_actions(mask, s, ped_id)
end
function POMDPStorm.safe_actions(mask::SafetyMask{PedMDP, PedMDPAction},s::UrbanState, ped_id=101)    
    s_mdp = get_pedmdp_state(mask.mdp, s, ped_id)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    # compute risk vector
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        si = state_index(mask.mdp, ss)
        p_sa += itp_weights[i]*mask.risk_mat[si,:]
    end
    safe_acts = PedMDPAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    if maximum(p_sa) <= mask.threshold
        push!(safe_acts, mask.actions[indmax(p_sa)])
    else
        for (j, a) in enumerate(mask.actions)
            if p_sa[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    return safe_acts
end

function POMDPToolbox.action_info{M}(policy::MaskedEpsGreedyPolicy{M}, s)
    return action(policy, s), safe_actions(policy.mask, s)
end

function animate_states(pomdp::UrbanPOMDP, states::Vector{UrbanState}, actions::Vector{UrbanAction}, mask::SafetyMask;
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 17.))
    duration = length(states)*pomdp.ΔT
    fps = Int(1/pomdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = states[frame_index] #state2scene(mdp, states[frame_index])
        safe_acts =[a.acc for a in safe_actions(mask, states[frame_index])]
        return AutoViz.render(scene,
                pomdp.env,
                cat(1, overlays, InterpolationOverlay(mask.mdp, pomdp),
                                 TextOverlay(text = ["v: $(get_ego(scene).state.v)"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,6.),
                                            incameraframe=true),
                                 TextOverlay(text = ["Acc: $(actions[frame_index].acc)"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,8.),
                                            incameraframe=true),
                                TextOverlay(text = ["Available Actions: $safe_acts"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,10.),
                                            incameraframe=true),
                                TextOverlay(text = ["step: $frame_index"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,4.),
                                            incameraframe=true)),
                cam=cam,
                car_colors=get_colors(scene))
    end
    return duration, fps, render_states
end

