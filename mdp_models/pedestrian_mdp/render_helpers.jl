"""
convert to AutomotiveDrivingModels.Scene
"""
function state2scene(mdp::PedMDP, s::PedMDPState, ped_type::VehicleDef = mdp.ped_type)
    scene = Scene()
    push!(scene, Vehicle(s.ego, mdp.ego_type, EGO_ID))
    push!(scene, Vehicle(s.ped, mdp.ped_type, 2))
    return scene
end

function animate_states(mdp::PedMDP, states::Vector{PedMDPState}, actions::Vector{PedMDPAction};
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 17.))
    duration = length(states)*mdp.ΔT
    fps = Int(1/mdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = state2scene(mdp, states[frame_index])
        return AutoViz.render(scene,
                mdp.env,
                cat(1, overlays, TextOverlay(text = ["Acc: $(actions[frame_index].acc)"],
                                            font_size=20,
                                            pos=VecE2(0.,8.),
                                            incameraframe=true)),
                cam=cam,
                car_colors=get_colors(scene))
    end
    return duration, fps, render_states
end


function animate_states(mdp::PedMDP, states::Vector{PedMDPState}, actions::Vector{PedMDPAction}, mask::SafetyMask;
                        overlays=SceneOverlay[IDOverlay()],
                        cam=StaticCamera(VecE2(0, -5.), 17.))
    duration = length(states)*mdp.ΔT
    fps = Int(1/mdp.ΔT)    
    function render_states(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        scene = state2scene(mdp, states[frame_index])
        safe_acts =[a.acc for a in safe_actions(mask, states[frame_index])]
        return AutoViz.render(scene,
                mdp.env,
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