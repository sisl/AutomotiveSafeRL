N_PROCS=56
addprocs(N_PROCS)
@everywhere begin 
    using POMDPs
    using DiscreteValueIteration 
    using AutomotivePOMDPs 
    include("myfile.jl")
    pomdp = MyPOMDP()
end 

solver = ParallelValueIterationSolver(n_procs=N_PROCS, max_iterations=2, belres=1e-4, include_Q=true, verbose=true)

vi_policy = solve(solver, mdp)
qmdp_policy = AlphaVectorPolicy(vi_policy)

# save policy!
