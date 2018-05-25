#=
Define a transition model for the car MDP problem
=#

function POMDPs.transition(mdp::CarMDP, s::CarMDPState, a::CarMDPAction)
    # ego transition
    acc = LonAccelDirection(a.acc, 4) #TODO paremeterize
    ego_p = propagate(s.ego, acc, mdp.env.roadway, mdp.ΔT)
    # interpolate ego_p in discrete space
    ego_v_space = get_car_vspace(mdp.env, mdp.vel_res)
    ego_ps, ego_probs = interpolate_state(ego_p, ego_v_space)
    # car transition
    car_probs = Float64[]
    car_routes = Vector{Lane}[]
    if s.car.posG == mdp.off_grid
        car_ps, car_routes = car_starting_states(mdp)
        push!(car_ps, get_off_the_grid(mdp))
        push!(car_routes, SVector{0, Lane}())
        car_probs = ones(length(car_ps))
        car_probs[end] = 1-mdp.car_birth
        car_probs[1:end-1] = mdp.car_birth/(length(car_ps)-1)
        @assert sum(car_probs) ≈ 1.0
    else
        # update model
        mdp.car_model.route = s.route # s.route is a static vector but not car_model.route
        lane = get_lane(mdp.env.roadway, s.car)
        set_direction!(mdp.car_model, lane, mdp.env.roadway)
        car_actions, car_actions_probs = get_distribution(mdp.car_model)
        car_ps = VehicleState[]
        for (i, car_a) in enumerate(car_actions)
            p_a = car_actions_probs[i]
            car_p = propagate(s.car, car_a, mdp.env.roadway, mdp.ΔT)
            if car_p.posF.s >= get_end(get_lane(mdp.env.roadway, car_p)) && isempty(get_lane(mdp.env.roadway, car_p).exits)
                car_pp = get_off_the_grid(mdp)
                index_car_pp = find(x -> x==car_pp, car_ps)
                if isempty(index_car_pp)
                    push!(car_ps, car_pp)
                    push!(car_probs, p_a)
                else
                    car_probs[index_car_pp] += p_a
                end
                car_routes = fill(s.route, length(car_ps))
                continue
            end
            # interpolate car_p in discrete space
            car_v_space = get_car_vspace(mdp.env, mdp.vel_res)
            itp_car_ps, itp_car_weights = interpolate_state(car_p, car_v_space)
            for (j, car_pss) in enumerate(itp_car_ps)
                index_itp_state = find(x -> x==car_pss, car_ps)
                if isempty(index_itp_state)
                    push!(car_ps, car_pss)
                    push!(car_probs, itp_car_weights[j]*p_a)
                else
                    car_probs[index_itp_state] += itp_car_weights[j]*p_a
                end
            end
            car_routes = fill(s.route, length(car_ps))
        end
        @assert length(car_probs) == length(car_ps)
        @assert length(car_routes) == length(car_ps)
        normalize!(car_probs, 1)
    end

    # combine car and ego states
    states_p = Vector{CarMDPState}(length(ego_ps)*length(car_ps)) # future states
    probs_p = Vector{Float64}(length(ego_ps)*length(car_ps)) # proba of future states
    k = 1
    for (i, e) in enumerate(ego_ps)
        for (j, p) in enumerate(car_ps)
            crash =  is_colliding(Vehicle(e, mdp.ego_type, 1), Vehicle(p, mdp.car_type, 2))
            states_p[k] = CarMDPState(crash, e, p, SVector(car_routes[j]...))
            probs_p[k] = ego_probs[i]*car_probs[j]
            k += 1
        end
    end
    normalize!(probs_p, 1)
    return SparseCat(states_p, probs_p)
end
