#=
Define a transition model for the pedestrian MDP problem
=#

function POMDPs.transition(mdp::PedMDP, s::PedMDPState, a::PedMDPAction)
    # ego transition
    acc = LonAccelDirection(a.acc, 4) #TODO paremeterize
    ego_p = propagate(s.ego, acc, mdp.env.roadway, mdp.ΔT)
    # interpolate ego_p in discrete space
    ego_v_space = get_car_vspace(mdp.env, mdp.vel_res)
    ego_ps, ego_probs = interpolate_state(ego_p, ego_v_space)
    ped_probs = Float64[]
    # pedestrian transition
    if s.ped.posG == mdp.off_grid
        ped_ps = pedestrian_starting_states(mdp)
        push!(ped_ps, get_off_the_grid(mdp))
        ped_probs = ones(length(ped_ps))
        ped_probs[end] = 1-mdp.ped_birth
        ped_probs[1:end-1] = mdp.ped_birth/(length(ped_ps)-1)
        @assert sum(ped_probs) ≈ 1.0
    else
        ped_actions, ped_actions_probs = get_distribution(mdp.ped_model)
        ped_ps = VehicleState[]
        for (i, ped_a) in enumerate(ped_actions)
            p_a = ped_actions_probs[i]
            ped_p = propagate(s.ped, ped_a, mdp.env.roadway, mdp.ΔT)
            if (ped_p.posF.s >= get_end(get_lane(mdp.env.roadway, ped_p)) && isapprox(ped_p.posF.ϕ, 0.)) ||
               (isapprox(ped_p.posF.s, 0., atol=0.01) && isapprox(ped_p.posF.ϕ, float(pi)))
                ped_pp = get_off_the_grid(mdp)
                index_ped_pp = find(x -> x==ped_pp, ped_ps)
                if isempty(index_ped_pp)
                    push!(ped_ps, get_off_the_grid(mdp))
                    push!(ped_probs, p_a)
                else
                    ped_probs[index_ped_pp] += p_a
                end
                continue
            end
            # interpolate ped_p in discrete space
            ped_v_space = get_ped_vspace(mdp.env, mdp.vel_ped_res)
            itp_ped_ps, itp_ped_weights = interpolate_pedestrian(ped_p, ped_v_space)
            for (j, ped_pss) in enumerate(itp_ped_ps)
                index_itp_state = find(x -> x==ped_pss, ped_ps)
                if isempty(index_itp_state)
                    push!(ped_ps, ped_pss)
                    push!(ped_probs, itp_ped_weights[j]*p_a)
                else
                    ped_probs[index_itp_state] += itp_ped_weights[j]*p_a
                end
            end
        end
        @assert length(ped_probs) == length(ped_ps)
        normalize!(ped_probs, 1)
    end

    # combine ped and ego states
    states_p = Vector{PedMDPState}(length(ego_ps)*length(ped_ps)) # future states
    probs_p = Vector{Float64}(length(ego_ps)*length(ped_ps)) # proba of future states
    k = 1
    for (i, e) in enumerate(ego_ps)
        for (j, p) in enumerate(ped_ps)
            crash =  is_colliding(Vehicle(e, mdp.ego_type, 1), Vehicle(p, mdp.ped_type, 2))
            states_p[k] = PedMDPState(crash, e, p)
            probs_p[k] = ego_probs[i]*ped_probs[j]
            k += 1
        end
    end
    normalize!(probs_p, 1)
    # remove duplicates
    return SparseCat(states_p, probs_p)
end
