
struct SingleAgentTracker <: Updater
    single_pomdp::UrbanPOMDP
    models::Vector{Chain}
    pres_threshold::Float64
    agent_def::VehicleDef
end

struct SingleAgentBelief 
    predictions::Vector{Vector{Float64}}
    obs::Vector{Float64}
    single_pomdp::UrbanPOMDP
end

struct MultipleAgentsTracker <: Updater
    pomdp::UrbanPOMDP
    ref_trackers::Dict{Int64, SingleAgentTracker} # bank of uninitialized tracker mapping class to reference tracker
    single_trackers::Dict{Int64, SingleAgentTracker}
end

struct MultipleAgentsBelief
    single_beliefs::Dict{Int64, SingleAgentBelief}
    o::Vector{Float64}
end


function POMDPs.update(up::MultipleAgentsTracker, bold::MultipleAgentsBelief, a::UrbanAction, o::UrbanObs)
#     println("Updater keys: ", keys(up.single_trackers))
#     println("belief keys: ", keys(bold.single_beliefs))
    ego, car_map, ped_map, obs_map = split_o(o, up.pomdp)
    bnew = Dict{Int64, SingleAgentBelief}()
    updated_ids = Set{Int64}()
    for (i, veh_map) in enumerate((car_map, ped_map))
        class = i == 1 ? AgentClass.CAR : AgentClass.PEDESTRIAN
        for (vehid, veh) in veh_map # iterate through visible cars
            single_o = vcat(ego, veh, obs_map[1])
            if haskey(up.single_trackers, vehid)
                @assert haskey(up.single_trackers, vehid)
                bnew[vehid] = update(up.single_trackers[vehid], bold.single_beliefs[vehid], a, single_o)
                push!(updated_ids, vehid)
            else
                # start new tracker 
                up.single_trackers[vehid] = deepcopy(up.ref_trackers[class])
#                 println("Starting new tracker for vehicle: ", vehid)
                init_belief = SingleAgentBelief(Vector{Vector{Float64}}(), Vector{Float64}(), up.single_trackers[vehid].single_pomdp)
                bnew[vehid] = update(up.single_trackers[vehid], init_belief, a, single_o)
                push!(updated_ids, vehid)
            end
        end
    end
      
    
    # add absent ped and car for each obstacle
    for (obsid, obs) in obs_map
        absent_obs = vcat(ego, get_normalized_absent_state(up.pomdp, ego), obs)
        for class in (AgentClass.CAR, AgentClass.PEDESTRIAN)
            if class == AgentClass.CAR
                new_id = pomdp.max_cars + obsid + 1
            else
                new_id = 100 + pomdp.max_peds + obsid + 1
            end
            
            if haskey(up.single_trackers, new_id) && haskey(bold.single_beliefs, new_id)
                bnew[new_id] = update(up.single_trackers[new_id], bold.single_beliefs[new_id], a, absent_obs)
            else
                # start new tracker 
                up.single_trackers[new_id] = deepcopy(up.ref_trackers[class])
                Flux.reset!.(up.single_trackers[new_id].models)
                init_belief = SingleAgentBelief(Vector{Vector{Float64}}(), Vector{Float64}(), up.single_trackers[new_id].single_pomdp)
                bnew[new_id] = update(up.single_trackers[new_id], init_belief, a, absent_obs)
            end
            push!(updated_ids, new_id)
        end
    end
    
    for (oldid, _) in up.single_trackers
        absent_obs = vcat(ego, get_normalized_absent_state(up.pomdp, ego), obs_map[1]) 
        if oldid ∉ updated_ids
#             println("Vehicle $oldid disappeared! still tracking")
            init_belief = SingleAgentBelief(Vector{Vector{Float64}}(), Vector{Float64}(), up.single_trackers[oldid].single_pomdp)
            bnew[oldid] = update(up.single_trackers[oldid], init_belief, a, absent_obs)
        end
    end
    return MultipleAgentsBelief(bnew, o)
end

function POMDPs.update(up::SingleAgentTracker, bold::SingleAgentBelief, a::UrbanAction, o::Vector{Float64}) # observation should be consistent with up.pomdp
    n_models = length(up.models)
    predictions = Vector{Vector{Float64}}(undef, n_models)
    for (i,m) in enumerate(up.models)
        predictions[i] = process_single_entity_prediction(up.single_pomdp, m(o).data, o, up.pres_threshold)
    end
    return SingleAgentBelief(predictions, o, up.single_pomdp)        
end

function process_single_entity_prediction(pomdp::UrbanPOMDP, b::Vector{Float64}, o::Vector{Float64}, pres_threshold::Float64=0.5)
    n_features = pomdp.n_features
    b_ = zeros(3*n_features) # should be 8 + obstacles
    b_[1:4] = o[1:4] # ego state fully observable
    # get car state from b
    car_presence = b[5]
    if rand() <  car_presence && (pres_threshold < car_presence)
        b_[n_features+1:2*n_features] = b[1:4]
    else
        # absent
        b_[n_features+1:2*n_features] = normalized_off_the_grid_pos(pomdp, o[1], o[2])
    end
    b_[2*n_features + 1:end] = o[end - n_features+1:end]
    return b_
end

function create_pedcar_beliefs(pomdp::UrbanPOMDP, b::MultipleAgentsBelief)
    if isempty(b.o)
        return  Dict{NTuple{3, Int64}, PedCarRNNBelief}()
    end
    ego, car_map, ped_map, obs_map = split_o(b.o, pomdp)
    # println("pedestrian detected :", keys(ped_map))
    # println("car detected : ", keys(car_map))
    bkeys = collect(keys(b.single_beliefs))
    car_ids = bkeys[bkeys .< 100]
    ped_ids = bkeys[bkeys .> 100]
    pedcar_beliefs = Dict{Tuple{Int64, Int64, Int64}, PedCarRNNBelief}()
    for carid in car_ids
        if carid <= pomdp.max_cars + 1
            obsid = 1
            if !haskey(car_map, carid)
                car_o = get_normalized_absent_state(pomdp, ego)
            else
                car_o = car_map[carid]
            end
        elseif pomdp.max_cars + 1 < carid < 100
            obsid = carid - pomdp.max_cars - 1
            car_o = get_normalized_absent_state(pomdp, ego)
        end
        for pedid in ped_ids
            if pedid <= 100 + pomdp.max_peds
                obsid_ = 1
                if !haskey(ped_map, pedid)
                    ped_o = get_normalized_absent_state(pomdp, ego)
                else
                    ped_o = ped_map[pedid]
                end
            else
                obsid_ = pedid - 101 - pomdp.max_peds
                ped_o = get_normalized_absent_state(pomdp, ego)
            end
            if obsid != obsid_
                continue
            end          
            o = vcat(ego, car_o, ped_o, obs_map[obsid]) #obstacle does not matter here 
            n_pred = length(b.single_beliefs[carid].predictions)
            predictions = Vector{Vector{Float64}}(undef, n_pred)
            for i=1:n_pred
                car_pred = b.single_beliefs[carid].predictions[i][pomdp.n_features+1:2*pomdp.n_features]
                ped_pred = b.single_beliefs[pedid].predictions[i][pomdp.n_features+1:2*pomdp.n_features]
                predictions[i] = vcat(ego, car_pred, ped_pred) #naïve
            end
            # println("Adding ", carid, " ", pedid, " ", obsid)
            pedcar_beliefs[(carid, pedid, obsid)] = PedCarRNNBelief(predictions, o)
        end
    end                     
    return pedcar_beliefs
end

struct SingleAgentBeliefOverlay <: SceneOverlay
    b::SingleAgentBelief
    color::Colorant
end

SingleAgentBeliefOverlay(b::SingleAgentBelief;  color = MONOKAI["color4"]) = SingleAgentBeliefOverlay(b, color)

function AutoViz.render!(rendermodel::RenderModel, overlay::SingleAgentBeliefOverlay, scene::Scene, env::OccludedEnv)
    for pred in overlay.b.predictions
        bel = [veh for veh in obs_to_scene(overlay.b.single_pomdp, pred) if veh.id != EGO_ID]
        AutoViz.render!(rendermodel, GaussianSensorOverlay(sensor=overlay.b.single_pomdp.sensor, o=bel, color=overlay.color), scene, env) 
    end
end

struct MultipleAgentsBeliefOverlay <: SceneOverlay
    b::MultipleAgentsBelief
    color::Colorant
end

MultipleAgentsBeliefOverlay(b::MultipleAgentsBelief;color = MONOKAI["color4"]) = MultipleAgentsBeliefOverlay(b, color)

function AutoViz.render!(rendermodel::RenderModel, overlay::MultipleAgentsBeliefOverlay, scene::Scene, env::OccludedEnv)
    for (id, bel) in overlay.b.single_beliefs
        bel_overlay = SingleAgentBeliefOverlay(bel, color=overlay.color)
        AutoViz.render!(rendermodel, bel_overlay, scene, env)
    end
end

struct PedCarBeliefOverlay <: SceneOverlay
    pomdp::UrbanPOMDP
    b::PedCarRNNBelief
    color::Colorant
end

function AutoViz.render!(rendermodel::RenderModel, overlay::PedCarBeliefOverlay, scene::Scene, env::OccludedEnv)
    for pred in overlay.b.predictions 
#         bb, _ = process_prediction(overlay.pomdp, pred, overlay.b.obs)
        bel = [veh for veh in obs_to_scene(overlay.pomdp, pred) if veh.id != EGO_ID]
        bel_overlay = GaussianSensorOverlay(sensor=overlay.pomdp.sensor, o=bel, color=overlay.color)
        AutoViz.render!(rendermodel, bel_overlay, scene, env)
    end
end