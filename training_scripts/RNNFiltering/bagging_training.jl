using Flux
using Flux: truncate!, reset!, batchseq, @epochs, params
using POMDPs
using POMDPToolbox
using AutomotiveDrivingModels
using AutomotivePOMDPs
using AutomotiveSensors
using PedCar
using UniversalTensorBoard
using ArgParse
using JLD
using ProgressMeter
include("RNNFiltering.jl")
using RNNFiltering

s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
        help = "an integer for the MersenneTwister"
        arg_type = Int64
        default = 1
end
parsed_args = parse_args(ARGS, s)

## RNG SEED 
seed = parsed_args["seed"]
rng = MersenneTwister(seed)
srand(rng)

# # init continuous state mdp 
mdp = PedCarMDP(pos_res=2.0, vel_res=2., ped_birth=0.7, car_birth=0.7)
pomdp = UrbanPOMDP(env=mdp.env,
                    sensor = GaussianSensor(false_positive_rate=0.05, 
                                            pos_noise = LinearNoise(min_noise=0.5, increase_rate=0.05), 
                                            vel_noise = LinearNoise(min_noise=0.5, increase_rate=0.05)),
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=1, 
                   car_birth=0.7, 
                   ped_birth=0.7, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   ego_start=20,
                   ΔT=0.1)

policy = RandomPolicy(rng, pomdp, VoidUpdater())

## Generate data 
max_steps = 400
n_train = 3000
if !isfile("train10k_"*string(seed)*".jld")
    println("Generating $n_train examples of training data")
    train_X, train_Y = collect_set(pomdp, policy, max_steps, rng, n_train)
    save("train10k_"*string(seed)*".jld", "train_X", train_X, "train_Y", train_Y)
else
    println("Loading existing training data: "*"train10k_"*string(seed)*".jld")
    train_data = load("train10k_"*string(seed)*".jld")
    train_X, train_Y = train_data["train_X"], train_data["train_Y"]
end
n_val = 1000
if !isfile("val1k_"*string(seed)*".jld")
    println("Generating $n_val examples of validation data")
    val_X, val_Y = collect_set(pomdp, policy, max_steps, rng, n_val)
    save("val1k_"*string(seed)*".jld", "val_X", val_X, "val_Y", val_Y)
else
    println("Loading existing validation data: "*"val1k_"*string(seed)*".jld")
    val_data = load("val1k_"*string(seed)*".jld")
    val_X, val_Y = val_data["val_X"], val_data["val_Y"]
end

model_name = "model_"*string(seed)

input_length = n_dims(pomdp) 
output_length = n_dims(pomdp) - 4

model = Chain(Dense(input_length, 32, tanh),
              LSTM(32, 128),
              Dense(128, 62, tanh),
              Dense(62, output_length))

macro interrupts(ex)
  :(try $(esc(ex))
    catch e
      e isa InterruptException || rethrow()
      throw(e)
    end)
end

function loss(x, y)
    l = mean(Flux.mse.(model.(x), y))
    reset!(model)
    return l
end

function training!(loss, train_data, val_X, val_Y, opt, n_epochs::Int64; logdir::String="log/model/")
    set_tb_logdir(logdir)
    total_time = 0.
    grad_norm = 0.
    for ep in 1:n_epochs 
        epoch_time = @elapsed begin
            training_loss = 0.
            @showprogress for d in train_data 
                l = loss(d...)
                @interrupts Flux.back!(l)
                grad_norm = global_norm(params(model))
                opt()
                training_loss += l.tracker.data 
            end
        end
        # log 
        total_time += epoch_time
        training_loss /= n_epochs
        validation_loss = mean(loss.(val_X, val_Y)).tracker.data
        set_tb_step!(ep)
        @tb_log training_loss
        set_tb_step!(ep)
        @tb_log validation_loss
        set_tb_step!(ep)
        @tb_log grad_norm
        logg = @sprintf("%5d / %5d Train loss %0.3e |  Val loss %1.3e | Grad %2.3e | Epoch time (s) %2.1f | Total time (s) %2.1f",
                                ep, n_epochs, training_loss, validation_loss, grad_norm, epoch_time, total_time)
        println(logg)
    end 
end

optimizer = RMSProp(Flux.params(model), 1e-3)

n_epochs = 5
reset_tb_logs()
training!(loss, zip(train_X, train_Y), val_X, val_Y, optimizer, n_epochs, logdir="log/"*model_name)

weights = Tracker.data.(Flux.params(model))
@save model_name*".bson" model
@save "weights_"*match(r"\d+", model_name).match*".bson" weights 