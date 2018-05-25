# A bunch of discretization helpers

# discretize velocity levels:
function get_car_vspace(env::UrbanEnv, v_res::Float64)
    return 0.:v_res:env.params.speed_limit
end

function get_ped_vspace(env::UrbanEnv, v_res::Float64)
    return 0:v_res:2.0
end

# discretize longitudinal position
function get_discretized_lane(tag::LaneTag, roadway::Roadway, pos_res::Float64)
    lane = roadway[tag]
    return 0.:pos_res:get_end(lane)
end

# enumerate ego state space
function get_ego_route(env::UrbanEnv)
    # TODO make it dependent on the topology
    return (LaneTag(6,1), LaneTag(13, 1), LaneTag(14,1), LaneTag(2,1))
end

function get_ego_states(env::UrbanEnv, pos_res::Float64, v_res::Float64)
    states_vec = VehicleState[]
    for lane in get_ego_route(env)
        discrete_lane = get_discretized_lane(lane, env.roadway, pos_res)
        v_space = get_car_vspace(env, v_res)
        for v in v_space
            for s in discrete_lane
                ego = VehicleState(Frenet(env.roadway[lane], s), env.roadway, v)
                push!(states_vec, ego)
            end
        end
    end
    return states_vec
end

# enumerate pedestrian state space
function get_ped_lanes(env::UrbanEnv)
    # TODO do not hard code it, get it form the environment structure
    return (LaneTag(17, 1), LaneTag(18, 1), LaneTag(19, 1))
end

function get_ped_states(env::UrbanEnv, pos_res::Float64, v_res::Float64)
    states_vec = VehicleState[]
    for lane in get_ped_lanes(env)
        discrete_lane = get_discretized_lane(lane, env.roadway, pos_res)
        v_space = get_ped_vspace(env, v_res)
        for v in v_space
            for s in discrete_lane
                for phi in (0., float(pi))
                    ped = VehicleState(Frenet(env.roadway[lane], s, 0., phi), env.roadway, v)
                    push!(states_vec, ped)
                end
            end
        end
    end
    return states_vec
end

# enumerate other car state space
function get_car_lanes(env::UrbanEnv)
    lanes = get_lanes(env.roadway)
    car_lanes = LaneTag[]
    no_go_lanes = [LaneTag(6, 1), LaneTag(15, 1), LaneTag(16, 1), LaneTag(14, 1), LaneTag(13, 1)]
    for lane in lanes
        if !(lane.tag ∈ no_go_lanes ) && !(lane.tag ∈ get_ped_lanes(env))
            push!(car_lanes, lane.tag)
        end
    end
    return car_lanes
end

# for route in routes
# for lane in route
# for s in lane ...

function get_car_states(env::UrbanEnv, pos_res::Float64, v_res::Float64)
    states_vec = VehicleState[]
    for lane in get_car_lanes(env)
        discrete_lane = get_discretized_lane(lane, env.roadway, pos_res)
        v_space = get_car_vspace(env, v_res)
        for v in v_space
            for s in discrete_lane
                car = VehicleState(Frenet(env.roadway[lane], s), env.roadway, v)
                push!(states_vec, car)
            end
        end
    end
    return states_vec
end

function get_car_states(env::UrbanEnv, route::StaticVector, pos_res::Float64, v_res::Float64)
    states_vec = VehicleState[]
    for lane in route
        discrete_lane = get_discretized_lane(lane.tag, env.roadway, pos_res)
        v_space = get_car_vspace(env, v_res)
        for v in v_space
            for s in discrete_lane
                car = VehicleState(Frenet(lane, s), env.roadway, v)
                push!(states_vec, car)
            end
        end
    end
    return states_vec
end


# count state spaces
function n_ego_states(env::UrbanEnv, pos_res::Float64, v_res::Float64)
    N = 0
    nv = length(get_car_vspace(env, v_res))
    for lane in get_ego_route(env)
        N += nv * length(get_discretized_lane(lane, env.roadway, pos_res))
    end
    return N
end

function n_ped_states(env::UrbanEnv, pos_res::Float64, v_res::Float64)
    N = 0
    nv = length(get_ped_vspace(env, v_res))
    n_headings = 2
    for lane in get_ped_lanes(env)
        N += nv * n_headings * length(get_discretized_lane(lane, env.roadway, pos_res))
    end
    return N
end

function n_car_states(env::UrbanEnv, pos_res::Float64, v_res::Float64)
    N = 0
    nv = length(get_car_vspace(env, v_res))
    for lane in get_car_lanes(env)
        N += nv * length(get_discretized_lane(lane, env.roadway, pos_res))
    end
    return N
end

function n_car_states(env::UrbanEnv, route::AbstractVector, pos_res::Float64, v_res::Float64)
    N = 0
    nv = length(get_car_vspace(env, v_res))
    for lane in route
        N += nv * length(get_discretized_lane(lane.tag, env.roadway, pos_res))
    end
    return N
end

# state indexing!
function ego_state_index(env::UrbanEnv, ego::VehicleState, pos_res::Float64, v_res::Float64)
    # find lane index
    lanes = get_ego_route(env)
    lane = get_lane(env.roadway, ego)
    li = findfirst(lanes, lane.tag) #XXX possibly inefficient
    # find position index
    s_space = get_discretized_lane(lane.tag, env.roadway, pos_res)
    si = find_range_index(s_space, ego.posF.s)
    # find velocity index
    v_space = get_car_vspace(env, v_res)
    size_v = length(v_space)
    v = ego.v
    vi = find_range_index(v_space, v)
    # sub2ind magic
    egoi = sub2ind((length(s_space), length(v_space)), si, vi)
    # Lanes have different lengths
    for i=2:li
        size_s = length(get_discretized_lane(lanes[i-1], env.roadway, pos_res))
        egoi += size_s*size_v
    end
    return egoi
end

function car_state_index(env::UrbanEnv, car::VehicleState, pos_res::Float64, v_res::Float64)
    # find lane index
    lanes = get_car_lanes(env)
    lane = get_lane(env.roadway, car)
    li = findfirst(lanes, lane.tag) #XXX possibly inefficient
    # find position index
    s_space = get_discretized_lane(lane.tag, env.roadway, pos_res)
    si = find_range_index(s_space, car.posF.s)
    # find velocity index
    v_space = get_car_vspace(env, v_res)
    size_v = length(v_space)
    v = car.v
    vi = find_range_index(v_space, v)
    # sub2ind magic
    cari = sub2ind((length(s_space), length(v_space)), si, vi)
    # Lanes have different lengths
    for i=2:li
        size_s = length(get_discretized_lane(lanes[i-1], env.roadway, pos_res))
        cari += size_s*size_v
    end
    return cari
end

function car_state_index(env::UrbanEnv, car::VehicleState, route::StaticVector, pos_res::Float64, v_res::Float64)
    lane = get_lane(env.roadway, car)
    li = findfirst(route, lane)
    # position index
    # find position index
    s_space = get_discretized_lane(lane.tag, env.roadway, pos_res)
    si = find_range_index(s_space, car.posF.s)
    # find velocity index
    v_space = get_car_vspace(env, v_res)
    size_v = length(v_space)
    v = car.v
    vi = find_range_index(v_space, v)
    # sub2ind magic
    cari = sub2ind((length(s_space), length(v_space)), si, vi)
    # Lanes have different lengths
    for i=2:li
        size_s = length(get_discretized_lane(route[i-1].tag, env.roadway, pos_res))
        cari += size_s*size_v
    end
    return cari
end

function ped_state_index(env::UrbanEnv, ped::VehicleState, pos_res::Float64, v_res::Float64)
    # find lane index
    lanes = get_ped_lanes(env)
    lane = get_lane(env.roadway, ped)
    li = findfirst(lanes, lane.tag) #XXX possibly inefficient
    # find position index
    s_space = get_discretized_lane(lane.tag, env.roadway, pos_res)
    si = find_range_index(s_space, ped.posF.s)
    # heading 
    n_headings = 2
    phi_i = ped.posF.ϕ == 0. ? 1 : 2
    # find velocity index
    v_space = get_ped_vspace(env, v_res)
    size_v = length(v_space)
    v = ped.v
    vi = find_range_index(v_space, v)
    # sub2ind magic
    pedi = sub2ind((n_headings, length(s_space), length(v_space)), phi_i, si, vi)
    # Lanes have different lengths
    for i=2:li
        size_s = length(get_discretized_lane(lanes[i-1], env.roadway, pos_res))
        pedi += size_s*size_v*n_headings
    end
    return pedi
end

function find_range_index(r::Range{Float64}, s::Float64)
    return round(Int, ((s - first(r))/step(r) + 1))
end


# transition as SparseCat?
# ego transition is known
# crash states are terminal
# goal states are terminal


# enumerate all the possible car routes
function get_car_routes(env::UrbanEnv)
    #TODO implement a routing algorithm
    straight_from_left = SVector(env.roadway[LaneTag(1, 1)],
                                       env.roadway[LaneTag(7, 1)],
                                       env.roadway[LaneTag(2, 1)])

    left_from_left = SVector(env.roadway[LaneTag(1, 1)],
                                      env.roadway[LaneTag(9, 1)],
                                      env.roadway[LaneTag(10, 1)],
                                      env.roadway[LaneTag(5, 1)])

    straight_from_right = SVector(env.roadway[LaneTag(3, 1)],
                                       env.roadway[LaneTag(8, 1)],
                                       env.roadway[LaneTag(4, 1)])

    right_from_right = SVector(env.roadway[LaneTag(3, 1)],
                                       env.roadway[LaneTag(11, 1)],
                                       env.roadway[LaneTag(12, 1)],
                                       env.roadway[LaneTag(5, 1)])
    return SVector(straight_from_left, left_from_left, straight_from_right, right_from_right)
end

function get_possible_routes(lane::Lane, env::UrbanEnv)
    possible_routes = Int64[]
    sizehint!(possible_routes, 2)
    routes = get_car_routes(env)
    for i=1:length(routes)
        route = routes[i]
        if in(lane, route)
            push!(possible_routes, i)
        end
    end
    return possible_routes
end
