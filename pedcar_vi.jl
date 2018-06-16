@everywhere begin
    using POMDPs, POMDPToolbox, DiscreteValueIteration
    using AutomotivePOMDPs, AutomotiveDrivingModels
end

rng = MersenneTwister(1)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(pos_res=6.0, vel_res=3.)
# mdp = PedCarMDP()

solver = ParallelValueIterationSolver()

policy = solve(solver, mdp, verbose=true)