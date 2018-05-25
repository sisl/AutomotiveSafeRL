# a bunch of interpolation helpers
function interpolate_state(state::VehicleState, v_space::StepRangeLen)
    # interpolate longitudinal position and velocity
    if state.posG == mdp.off_grid
        return (state,), (1.0,)
    end
    lane = get_lane(mdp.env.roadway, state)
    l_space = get_discretized_lane(lane.tag, mdp.env.roadway, mdp.pos_res)
    grid = RectangleGrid(l_space, v_space)
    real_state = SVector{2, Float64}(state.posF.s, state.v)
    idx, weights = interpolants(grid, real_state)
    n_pts = length(idx)
    states = Vector{VehicleState}(n_pts)
    probs = zeros(n_pts)
    for i=1:n_pts
        sg, vg = ind2x(grid, idx[i])
        states[i] = VehicleState(Frenet(lane, sg), mdp.env.roadway, vg)
        probs[i] = weights[i]
    end
    return states, probs
end

# take into account heading as well
function interpolate_pedestrian(state::VehicleState, v_space::StepRangeLen)
    # interpolate longitudinal position and velocity
    if state.posG == mdp.off_grid
        return (state,), (1.0,)
    end
    lane = get_lane(mdp.env.ped_roadway, state)
    phi_space = SVector{2, Float64}(0., float(pi))
    l_space = get_discretized_lane(lane.tag, mdp.env.ped_roadway, mdp.pos_res)
    grid = RectangleGrid(l_space, v_space, phi_space)
    real_state = SVector{3, Float64}(state.posF.s, state.v, state.posF.ϕ)
    idx, weights = interpolants(grid, real_state)
    n_pts = length(idx)
    states = Vector{VehicleState}(n_pts)
    probs = zeros(n_pts)
    for i=1:n_pts
        sg, vg, phig = ind2x(grid, idx[i])
        states[i] = VehicleState(Frenet(lane, sg, 0., phig), mdp.env.ped_roadway, vg)
        probs[i] = weights[i]
    end
    return states, probs
end