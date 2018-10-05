function animate_states(pomdp::M, states::Vector{S}, actions::Vector{A}, mask::SafetyMask;
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 17.)) where {M,S,A}
    duration = length(states)*pomdp.ΔT
    fps = Int(1/pomdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = state2scene(mdp, states[frame_index])
        safe_acts =[a.acc for a in safe_actions(mask, states[frame_index])]
        return AutoViz.render(scene,
                pomdp.env,
                cat(1, overlays, TextOverlay(text = ["Acc: $(actions[frame_index].acc)"],
                                            font_size=20,
                                            pos=VecE2(0.,8.),
                                            incameraframe=true),
                                TextOverlay(text = ["Available Actions: $safe_acts"],
                                            font_size=20,
                                            pos=VecE2(0.,10.),
                                            incameraframe=true)),
                cam=cam,
                car_colors=get_colors(scene))
    end
    return duration, fps, render_states
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

function animate_states(pomdp::UrbanPOMDP, 
                        states::Vector{UrbanState}, 
                        actions::Vector{UrbanAction}, 
                        observations::Vector{Vector{Float64}},
                        safe_actions::Vector{Vector{UrbanAction}}, 
                        probas::Vector{Vector{Float64}}, 
                        routes::Vector{SVector{2, LaneTag}},
                        mask::SafetyMask;
                        interp=true,
                        obsviz=true,
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 17.))
    duration = length(states)*pomdp.ΔT
    fps = Int(1/pomdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = states[frame_index] #state2scene(mdp, states[frame_index])
        safe_acts = [a.acc for a in safe_actions[frame_index]]
        probs = [round(p, 5) for p in probas[frame_index]]
        r1, r2 = routes[frame_index]
        obs = obs_to_scene(pomdp, observations[frame_index])
        if !obsviz
            obs_overlay = SceneOverlay[]
        else
            obs_overlay = [GaussianSensorOverlay(sensor=pomdp.sensor, o=[veh for veh in obs if veh.id != EGO_ID])]
        end
        if !interp
            itp_overlay = SceneOverlay[]
        else
            itp_overlay=SceneOverlay[InterpolationOverlay(mask.mdp, pomdp, obs)]
        end       
        return AutoViz.render(scene,
                pomdp.env,
                cat(1, overlays,itp_overlay..., obs_overlay...,
                                TextOverlay(text = ["v: $(get_ego(scene).state.v)"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,4.),
                                            incameraframe=true),
                                 TextOverlay(text = ["Acc: $(actions[frame_index].acc)"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,6.),
                                            incameraframe=true),
                                TextOverlay(text = ["Available Actions: $safe_acts"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,10.),
                                            incameraframe=true),
                                TextOverlay(text = ["Action probas: $probs"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,8.),
                                            incameraframe=true),
                                TextOverlay(text = ["step: $frame_index"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,-5.),
                                            incameraframe=true),
                                TextOverlay(text = ["route: ($(r1.segment), $(r1.lane)), ($(r2.segment), $(r2.lane))"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,-7.),
                                            incameraframe=true)),
                cam=cam,
                car_colors=get_colors(scene))
    end
    return duration, fps, render_states
end

function animate_states(pomdp::UrbanPOMDP, states::Vector{UrbanState}, actions::Vector{UrbanAction}, safe_actions::Vector{Any};
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 17.))
    duration = length(states)*pomdp.ΔT
    fps = Int(1/pomdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = states[frame_index] #state2scene(mdp, states[frame_index])
        safe_acts = [a.acc for a in safe_actions[frame_index]]
        return AutoViz.render(scene,
                pomdp.env,
                cat(1, overlays, TextOverlay(text = ["v: $(get_ego(scene).state.v)"],
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

function animate_states(pomdp::UrbanPOMDP, states::Vector{UrbanState}, actions::Vector{UrbanAction}, safe_acts_hist::Vector{Any}, mask::JointMask;
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 16.),
                        interp=true)
    duration = length(states)*pomdp.ΔT
    fps = Int(1/pomdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = states[frame_index] #state2scene(mdp, states[frame_index])
        safe_acts = [a.acc for a in safe_acts_hist[frame_index]]
        ped_safe_acts = [a.acc for a in safe_actions(pomdp, mask.masks[1], scene)]
        itp_overlays = [InterpolationOverlay(mask.masks[i].mdp, pomdp, scene, mask.ids[i]) for i=1:length(mask.masks)]
        if !interp
            itp_overlays = SceneOverlay[]
        end
        return AutoViz.render(scene,
                pomdp.env,
                cat(1, overlays,itp_overlays...,
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
                                TextOverlay(text = ["Available Actions (Pedestrian only): $ped_safe_acts"],
                                            font_size=20,
                                            pos=VecE2(pomdp.env.params.x_min + 3.,12.),
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


function render_interpolated_states(mdp::CarMDP, pomdp::UrbanPOMDP, scene::Scene, car_id=2)
    sp = deepcopy(scene)
    s = get_mdp_state(mdp, pomdp, scene, car_id)
    vspace = get_car_vspace(mdp.env, mdp.vel_res)
    itp_car, itp_car_w = interpolate_state(mdp, s.car, vspace)
    itp_ego, itp_ego_w = interpolate_state(mdp, s.ego, vspace)
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

mutable struct InterpolationOverlay{M} <: SceneOverlay where {M <: Union{CarMDP, PedMDP, PedCarMDP}}
    verbosity::Int
    color::Colorant
    font_size::Int
    id::Int
    mdp::M
    pomdp::UrbanPOMDP
    obs::Scene

    function InterpolationOverlay(mdp::M, pomdp::UrbanPOMDP, obs::Scene, id::Int=2;
        verbosity::Int=1,
        color::Colorant=colorant"white",
        font_size::Int=20,
        ) where {M <: Union{CarMDP, PedMDP, PedCarMDP}}

        new{M}(verbosity, color, font_size, id, mdp, pomdp, obs)
    end
end

function AutoViz.render!(rendermodel::RenderModel, overlay::InterpolationOverlay{PedCarMDP}, scene::Scene, env::OccludedEnv)
    transparency = 0.5
    s_mdp = get_mdp_state(mask.mdp, pomdp, overlay.obs, 101, 2)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    for ss in itp_states 
        ci = ss.car
        x, y, θ, v = ci.posG.x, ci.posG.y, ci.posG.θ, ci.v 
        color = RGBA(75.0 /255, 66.0/255, 244.0/255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.car_type.length, overlay.mdp.car_type.width, color, color, RGBA(1.,1.,1.,transparency)))
        pi = ss.ped
        x, y, θ, v = pi.posG.x, pi.posG.y, pi.posG.θ, pi.v 
        color = RGBA(75.0 /255, 66.0/255, 244.0/255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ped_type.length, overlay.mdp.ped_type.width, color, color, RGBA(1.,1.,1.,transparency)))
        ei = ss.ego 
        x, y, θ, v = ei.posG.x, ei.posG.y, ei.posG.θ, ei.v 
        color = RGBA(75.0 /255, 66.0/255, 244.0/255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ego_type.length, overlay.mdp.ego_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end
end


function AutoViz.render!(rendermodel::RenderModel, overlay::InterpolationOverlay{CarMDP}, scene::Scene, env::OccludedEnv)
    transparency = 0.5
    s = get_mdp_state(overlay.mdp, overlay.pomdp, scene, overlay.id)
    vspace = get_car_vspace(overlay.mdp.env, overlay.mdp.vel_res)
    itp_car, itp_car_w = interpolate_state(overlay.mdp, s.car, vspace)
    itp_ego, itp_ego_w = interpolate_state(overlay.mdp, s.ego, vspace)
    for ci in itp_car
        x, y, θ, v = ci.posG.x, ci.posG.y, ci.posG.θ, ci.v 
        color = RGBA(75.0/255, 66.0/255, 244.0/255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.car_type.length, overlay.mdp.car_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end
    for ei in itp_ego
         x, y, θ, v = ei.posG.x, ei.posG.y, ei.posG.θ, ei.v 
        color = RGBA(75.0/255, 66.0/255, 244.0/255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ego_type.length, overlay.mdp.ego_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end    
end

function AutoViz.render!(rendermodel::RenderModel, overlay::InterpolationOverlay{PedMDP}, scene::Scene, env::OccludedEnv)
    transparency = 0.5
    s = get_mdp_state(overlay.mdp, scene, overlay.id)
    ped_v = get_ped_vspace(overlay.mdp.env, overlay.mdp.vel_ped_res) 
    itp_ped, itp_ped_w = interpolate_pedestrian(overlay.mdp, s.ped, ped_v)
    ego_v = get_car_vspace(overlay.mdp.env, overlay.mdp.vel_res)
    itp_ego, itp_ego_w = interpolate_state(overlay.mdp, s.ego, ego_v)
    for pi in itp_ped
        x, y, θ, v = pi.posG.x, pi.posG.y, pi.posG.θ, pi.v 
        color = RGBA(75.0/255, 66.0/255, 244.0/255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ped_type.length, overlay.mdp.ped_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end
    for ei in itp_ego
         x, y, θ, v = ei.posG.x, ei.posG.y, ei.posG.θ, ei.v 
        color = RGBA(75.0/255, 66.0/255, 244.0/255, transparency)
        add_instruction!(rendermodel, render_vehicle, (x, y, θ, overlay.mdp.ego_type.length, overlay.mdp.ego_type.width, color, color, RGBA(1.,1.,1.,transparency)))
    end    
end

# function animate_states(mdp::PedMDP, states::Vector{PedMDPState}, actions::Vector{PedMDPAction}, mask::SafetyMask;
#                         overlays=SceneOverlay[IDOverlay()],
#                         cam=StaticCamera(VecE2(0, -5.), 17.))
#     duration = length(states)*mdp.ΔT
#     fps = Int(1/mdp.ΔT)    
#     function render_states(t, dt)
#         frame_index = Int(floor(t/dt)) + 1
#         scene = state2scene(mdp, states[frame_index])
#         safe_acts =[a.acc for a in safe_actions(mask, states[frame_index])]
#         return AutoViz.render(scene,
#                 mdp.env,
#                 cat(1, overlays, TextOverlay(text = ["Acc: $(actions[frame_index].acc)"],
#                                             font_size=20,
#                                             pos=VecE2(0.,8.),
#                                             incameraframe=true),
#                                 TextOverlay(text = ["Available Actions: $safe_acts"],
#                                             font_size=20,
#                                             pos=VecE2(0.,10.),
#                                             incameraframe=true)),
#                 cam=cam,
#                 car_colors=get_colors(scene))
#     end
#     return duration, fps, render_states
# end
