using Flux
using StaticArrays
using ProgressMeter
using POMDPs
using POMDPToolbox
using AutomotiveDrivingModels
using AutomotivePOMDPs
using AutomotiveSensors
using PedCar
using Flux: truncate!, reset!, batchseq, @epochs, params
using UniversalTensorBoard
using BSON: @save

function loss(x, y)
    l = mean(Flux.mse.(model.(x), y))
    reset!(model)
    return l
end

function global_norm(W)
    return maximum(maximum(abs.(w.grad)) for w in W)
end

function set_tb_step!(t)
    UniversalTensorBoard.default_logging_session[].global_step =  t
end

function training!(loss, train_data, opt, n_epochs::Int64; logdir::String="log/model/")
    set_tb_logdir(logdir)
    total_time = 0.
    grad_norm = 0.
    for ep in 1:n_epochs 
        epoch_time = @elapsed begin
        training_loss = 0.
        for d in train_data 
            l = loss(d...)
            Flux.back!(l)
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


# init continuous state mdp 
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
                   ΔT=0.5)

rng = MersenneTwister(1)
policy = RandomPolicy(rng, pomdp, VoidUpdater())


model_name = "model_4"

input_length = n_dims(pomdp) 
output_length = n_dims(pomdp) - 4

model = Chain(LSTM(input_length, 128),
              Dense(128, output_length))


optimizer = RMSProp(Flux.params(model), 1e-2)

n_epochs = 20
reset_tb_logs()
training!(loss, zip(train_X, train_Y), optimizer, n_epochs, logdir="log/"*model_name)

weights = Tracker.data.(Flux.params(model))
@save model_name*".bson" model
@save "weights_"*match(r"\d+", model_name).match*".bson" weights 