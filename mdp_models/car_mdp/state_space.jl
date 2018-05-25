#=
Define a discrete state space for the car MDP problem
=#

function POMDPs.n_states(mdp::CarMDP)
    n_ego = n_ego_states(mdp.env, mdp.pos_res, mdp.vel_res)
    routes = get_car_routes(mdp.env)
    n_cars = 0
    for route in routes
        n_cars += n_car_states(mdp.env, route, mdp.pos_res, mdp.vel_res)
    end
    return n_ego*(n_cars + 1) # do not forget absent state
end

function POMDPs.states(mdp::CarMDP)
    state_space = CarMDPState[]
    ego_states = get_ego_states(mdp.env, mdp.pos_res, mdp.vel_res)
    routes = get_car_routes(mdp.env)
    for route in routes
        for ego in ego_states
            car_states = get_car_states(mdp.env, route, mdp.pos_res, mdp.vel_res)
            for car in car_states
                crash =  is_colliding(Vehicle(ego, mdp.ego_type, 1), Vehicle(car, mdp.car_type, 2))
                # enumerate all possible routes
                lane = get_lane(mdp.env.roadway, car)
                push!(state_space, CarMDPState(crash, ego, car, route))
            end
        end
    end
    for ego in ego_states
        # add absent states
        push!(state_space, CarMDPState(false, ego, get_off_the_grid(mdp), SVector{0, Lane}()))
    end
    return state_space
end


function POMDPs.state_index(mdp::CarMDP, s::CarMDPState)
    n_ego = n_ego_states(mdp.env, mdp.pos_res, mdp.vel_res)
    n_car = n_car_states(mdp.env, s.route, mdp.pos_res, mdp.vel_res)
    routes = get_car_routes(mdp.env)
    # step 1: find ego_index
    ego_i = ego_state_index(mdp.env, s.ego, mdp.pos_res, mdp.vel_res)
    # handle off the grid case
    if s.car.posG == mdp.off_grid
        si = 0
        for route in routes
            si += n_ego * n_car_states(mdp.env, route, mdp.pos_res, mdp.vel_res)
        end
        si += ego_i
    else
        # step 2: find route_index
        route_i = 0
        for (i, route) in enumerate(routes)
            if [l.tag for l in route] == [l.tag for l in s.route]
                route_i = i
            end
        end
        # step 3: find car_index in car states
        car_i = car_state_index(mdp.env, s.car, s.route, mdp.pos_res, mdp.vel_res)
        # sub2ind magic
        si = sub2ind((n_car, n_ego), car_i, ego_i)

        for i=2:route_i
            size_r = n_ego * n_car_states(mdp.env, routes[i-1], mdp.pos_res, mdp.vel_res)
            si += size_r
        end
    end
    return si
end

function POMDPs.initial_state_distribution(mdp::CarMDP)
    ego = initial_ego_state(mdp)
    init_car_states, init_car_routes = initial_car_state_distribution(mdp)
    init_states = Vector{CarMDPState}(length(init_car_states))
    for i=1:length(init_car_states)
        car = init_car_states[i]
        route = init_car_routes[i]
        crash = is_colliding(Vehicle(ego, mdp.ego_type, 1), Vehicle(car, mdp.car_type, 2))
        init_states[i] = CarMDPState(crash, ego, car, route)
    end
    # uniform
    probs = ones(length(init_car_states))
    normalize!(probs, 1)
    return SparseCat(init_states, probs)
end

function initial_ego_state(mdp::CarMDP)
    lanes = get_ego_route(mdp.env)
    posF = Frenet(mdp.env.roadway[lanes[1]], mdp.ego_start)
    v0 = 0.
    return VehicleState(posF, mdp.env.roadway, v0)
end

function initial_car_state_distribution(mdp::CarMDP)
    routes = get_car_routes(mdp.env)
    init_car_routes = []
    init_car_states = VehicleState[]
    for route in routes 
        car_states = get_car_states(mdp.env, route, mdp.pos_res, mdp.vel_res)
        for cs in car_states 
            push!(init_car_states, cs)
            push!(init_car_routes, route)
        end
    end
    push!(init_car_states, get_off_the_grid(mdp))
    push!(init_car_routes, SVector{0, Lane}())
    return init_car_states, init_car_routes
end

function initial_car_state(mdp::CarMDP, rng::AbstractRNG)
    init_car_states, init_car_routes = initial_car_state_distribution(mdp)
    return rand(rng, init_car_states), rand(rng, init_car_routes)
end

function POMDPs.initial_state(mdp::CarMDP, rng::AbstractRNG)
    # routes = get_car_routes(mdp.env)
    # route = rand(rng, routes)
    # car = rand(rng, get_car_states(mdp.env, route, mdp.pos_res, mdp.vel_res))
    # ego = initial_ego_state(mdp)
    # crash =  is_colliding(Vehicle(ego, mdp.ego_type, 1), Vehicle(car, mdp.car_type, 2))
    return rand(rng, initial_state_distribution(mdp))
end

function car_starting_states(mdp::CarMDP, min_speed::Float64 = 6.0)
    # list of car starting states
    routes = get_car_routes(mdp.env)
    v_space = min_speed:mdp.vel_res:mdp.env.params.speed_limit 
    N_states = 0
    for route in routes 
        N_states += length(v_space)
    end
    car_states = Vector{VehicleState}(N_states)
    start_routes = Vector{Vector{Lane}}(N_states)
    i = 1
    for route in routes
        lane = route[1]
        for v in v_space
            car_states[i] = VehicleState(Frenet(lane, 0.), mdp.env.roadway, v)
            start_routes[i] = route
            i += 1
        end
    end
    return car_states, start_routes
end

function interpolate_state(mdp::CarMDP, s::CarMDPState)
    throw("NOT IMPLEMENTED")
end