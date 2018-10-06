function interpolate_state(mdp::CarMDP, s::CarMDPState)
    # interpolate s in the grid
    vspace = get_car_vspace(mdp.env, mdp.vel_res)
    itp_car, itp_car_w = interpolate_state(s.car, vspace)
    itp_ego, itp_ego_w = interpolate_state(s.ego, vspace)
    itp_states = Vector{CarMDPState}(length(itp_ego)*length(itp_car))
    itp_w = Vector{Float64}(length(itp_states))
    k = 1
    for (i, c) in enumerate(itp_car)
        for (j, e) in enumerate(itp_ego)
            crash = is_colliding(Vehicle(c, mdp.car_type, 0), Vehicle(e, mdp.ego_type, 1))
            itp_states[k] = CarMDPState(crash, e, c, s.route)
            itp_w[k] = itp_car_w[i]*itp_ego_w[j]
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


function get_carmdp_state(mdp::CarMDP, pomdp::UrbanPOMDP, s::Scene, car_id = 2)
    car_i = findfirst(s, car_id)
    car = Vehicle(get_off_the_grid(mask.mdp), mdp.car_type, car_id)
    if car_i != 0
        car = s[car_i]
    end
    ego = get_ego(s)
    route = SVector{0, Lane}()
    if haskey(pomdp.models, car_id)
        route = SVector(pomdp.models[car_id].navigator.route...)
    end
    e_state = VehicleState(ego.state.posG, car_roadway(mdp.env), ego.state.v)
    c_state = VehicleState(car.state.posG, mdp.env.roadway, car.state.v)
    return CarMDPState(is_colliding(ego, car), e_state, c_state, route)
end

function render_interpolated_states(mdp::CarMDP, pomdp::UrbanPOMDP, scene::Scene, car_id=2)
    sp = deepcopy(scene)
    s = get_carmdp_state(mdp, pomdp, scene, car_id)
    vspace = get_car_vspace(mdp.env, mdp.vel_res)
    itp_car, itp_car_w = interpolate_state(s.car, vspace)
    itp_ego, itp_ego_w = interpolate_state(s.ego, vspace)
    colors = Dict{Int64, Colorant}(1=>COLOR_CAR_EGO, car_id=>COLOR_CAR_OTHER)
    cid = car_id+1
    for ci in itp_car
        if ci.posG != mdp.off_grid
            push!(sp, Vehicle(ci, mdp.car_type, cid))
            colors[cid] = MONOKAI["color2"]
            cid += 1            
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
    mdp::CarMDP
    pomdp::UrbanPOMDP

    function InterpolationOverlay(mdp::CarMDP, pomdp::UrbanPOMDP, id::Int=2;
        verbosity::Int=1,
        color::Colorant=colorant"white",
        font_size::Int=20,
        )

        new(verbosity, color, font_size, id, mdp, pomdp)
    end
end

function AutoViz.render!(rendermodel::RenderModel, overlay::InterpolationOverlay, scene::Scene, env::OccludedEnv)
    transparency = 0.5
    s = get_carmdp_state(overlay.mdp, overlay.pomdp, scene, overlay.id)
    vspace = get_car_vspace(overlay.mdp.env, overlay.mdp.vel_res)
    itp_car, itp_car_w = interpolate_state(s.car, vspace)
    itp_ego, itp_ego_w = interpolate_state(s.ego, vspace)
    for ci in itp_car
        x, y, θ, v = ci.posG.x, ci.posG.y, ci.posG.θ, ci.v 
        color = RGBA(75./255, 66./255, 244./255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.car_type.length, overlay.mdp.car_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end
    for ei in itp_ego
         x, y, θ, v = ei.posG.x, ei.posG.y, ei.posG.θ, ei.v 
        color = RGBA(75./255, 66./255, 244./255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ego_type.length, overlay.mdp.ego_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end    
end

# XXX uses global variable POMDP
function POMDPStorm.safe_actions(mask::SafetyMask{CarMDP, CarMDPAction}, o::UrbanObs, ped_id=2)
    s = obs_to_scene(pomdp, o)
    return safe_actions(pomdp, mask, s, ped_id)
end
function POMDPStorm.safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{CarMDP, CarMDPAction}, s::UrbanState, car_id=2)    
    s_mdp = get_carmdp_state(mask.mdp, pomdp, s)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    # compute risk vector
    # si = stateindex(mdp, itp_states[indmax(itp_weights)])
    # p_sa = mask.risk_mat[si, :]
#     p_sa_itp = zeros(length(itp_states), n_actions(mask.mdp))
#     for (i, ss) in enumerate(itp_states)
#         si = stateindex(mask.mdp, ss)
#         p_sa_itp[i, :] += itp_weights[i]*mask.risk_mat[si,:]
#     end
#     p_sa = minimum(p_sa_itp, 1)
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        si = stateindex(mask.mdp, ss)
        p_sa += itp_weights[i]*mask.risk_mat[si,:]
    end
    safe_acts = CarMDPAction[]
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
#     println("coucou ")
#     println("Safe acts $([a.acc for a in safe_acts])")
    return safe_acts
end

function POMDPToolbox.action_info{M}(policy::MaskedEpsGreedyPolicy{M}, s)
    return action(policy, s), nothing#safe_actions(policy.mask, s)
end

function animate_states(pomdp::UrbanPOMDP, states::Vector{UrbanState}, actions::Vector{UrbanAction}, mask::SafetyMask;
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 17.))
    duration = length(states)*pomdp.ΔT
    fps = Int(1/pomdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = states[frame_index] #state2scene(mdp, states[frame_index])
        safe_acts =[a.acc for a in safe_actions(pomdp, mask, states[frame_index])]
        return AutoViz.render(scene,
                pomdp.env,
                cat(1, overlays,InterpolationOverlay(mask.mdp, pomdp),
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