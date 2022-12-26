include("helpers/data_helper.jl")
include("helpers/train_helper.jl")


# Construct an O-NET
begin
    using Lux
    O_NET = Lux.Chain(Lux.Dense(3, 20, tanh),
                                Lux.Dense(20, 10, tanh),
                                Lux.Dense(10, 3))

    using Random
    rng = Random.default_rng()
    θ_O_NET, st_O_NET = Lux.setup(rng, O_NET)
end


# Construct a Structured ODE Neural Network
begin
    using Lux
    Structured_ODE_NN = Lux.Chain(Lux.Dense(1, 20, tanh),
                                  Lux.Dense(20, 10, tanh),
                                  Lux.Dense(10, 2))

    using Random
    rng = Random.default_rng()
    θ_Structured_ODE_NN, st_Structured_ODE_NN = Lux.setup(rng, Structured_ODE_NN)
end


# Generate ODE data of an isothermal damped harmonic oscillator
begin
    using Main.DataHelper: ODEfunc_idho, ODESolver
    # mass m, spring compliance c, the damping coefficient d and the environment temperature θ_0
    params = [2, 1, 0.5, 300]
    # initial state
    initial_state = [1.0, 1.0, 0.2]
    # Generate data set
    time_span = (0.0, 9.9)
    time_step = range(0.0, 9.9, 100)
    ode_data = ODESolver(ODEfunc_idho, params, initial_state, time_span, time_step)
end


# Generate predict data
begin
    # Generate O-NET data
    using Main.DataHelper: NeuralODESolver
    function NeuralODE_O_NET(dz, z, θ_O_NET, t)
        dz[1] = O_NET(z, θ_O_NET, st_O_NET)[1][1]
        dz[2] = O_NET(z, θ_O_NET, st_O_NET)[1][2]
        dz[3] = O_NET(z, θ_O_NET, st_O_NET)[1][3]
    end
    predict_data_O_NET = NeuralODESolver(NeuralODE_O_NET, θ_O_NET, initial_state, time_span, time_step)
  
    # Generate Structured ODE Neural Network data
    m = params[1]
    c = params[2]
    function NeuralODE_Structured_ODE_NN(dz, z, θ_Structured_ODE_NN, t)
        q = z[1]
        p = z[2]
        s_e = z[3]
        v = p/m
        dz[1] = v
        dz[2] = -q/c + Structured_ODE_NN([v], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][1]
        dz[3] = Structured_ODE_NN([v^2], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][2]
    end
    predict_data_Structured_ODE_NN = NeuralODESolver(NeuralODE_Structured_ODE_NN, θ_Structured_ODE_NN, initial_state, time_span, time_step)
end




############
# Training #
############

# Construct loss function, callback function, dataloader and optimization function
begin
    using Main.TrainInterface: SolveIVP, OptFunction
    using Flux: DataLoader

    function loss_function_O_NET(θ, batch_data, batch_timesteps)
        pred_data = SolveIVP(NeuralODE_O_NET, θ, initial_state, batch_timesteps)
        loss = sum((batch_data[1,:] .- pred_data[1,:]) .^ 2 +
                   (batch_data[2,:] .- pred_data[2,:]) .^ 2 +
                   (batch_data[3,:] .- pred_data[3,:]) .^ 2)
        return loss, pred_data
    end

    function loss_function_Structured_ODE_NN(θ, batch_data, batch_timesteps)
        pred_data = SolveIVP(NeuralODE_Structured_ODE_NN, θ, initial_state, batch_timesteps)
        loss = sum((batch_data[1,:] .- pred_data[1,:]) .^ 2 +
                   (batch_data[2,:] .- pred_data[2,:]) .^ 2 +
                   (batch_data[3,:] .- pred_data[3,:]) .^ 2)
        return loss, pred_data
    end

    callback_O_NET = function(θ, loss, pred_data)
        println(loss_function_O_NET(θ, ode_data, time_step)[1])
        return false
    end

    callback_Structured_ODE_NN = function(θ, loss, pred_data)
        println(loss_function_Structured_ODE_NN(θ, ode_data, time_step)[1])
        return false
    end

    dataloader = DataLoader((ode_data, time_step), batchsize = 100)

    begin
        optf_O_NET = OptFunction(loss_function_O_NET)
        optf_Structured_ODE_NN = OptFunction(loss_function_Structured_ODE_NN)
    end
end


# Repeat training for the O-NET
begin
    using Main.TrainInterface: LuxTrain
    α = 0.001
    epochs = 100
    θ_O_NET = LuxTrain(optf_O_NET, θ_O_NET, α, epochs, dataloader, callback_O_NET)
end


# Repeat training for the Structured ODE Neural Network
begin
    using Main.TrainInterface: LuxTrain
    α = 0.001
    epochs = 100
    θ_Structured_ODE_NN = LuxTrain(optf_Structured_ODE_NN, θ_Structured_ODE_NN, α, epochs, dataloader, callback_Structured_ODE_NN)
end


# Save the parameters and models of O-NET
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "parameters", "params_experiment_idho_O_NET.jld2")
    JLD2.save(path, "params_experiment_idho_O_NET", θ_O_NET)
    path = joinpath(@__DIR__, "models", "experiment_idho_O_NET.jld2")
    JLD2.save(path, "experiment_idho_O_NET", O_NET, "st", st_O_NET)
end


# Save the parameters and models of Structured ODE Neural Network
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "parameters", "params_experiment_idho_structured_ODE_NN.jld2")
    JLD2.save(path, "params_experiment_idho_structured_ODE_NN", θ_Structured_ODE_NN)
    path = joinpath(@__DIR__, "models", "experiment_idho_structured_ODE_NN.jld2")
    JLD2.save(path, "experiment_idho_structured_ODE_NN", Structured_ODE_NN, "st", st_Structured_ODE_NN)
end




##############
# Evaluation #
##############

# Load the parameters and models
begin
    begin
        using JLD2, Lux
        path = joinpath(@__DIR__, "parameters", "params_experiment_idho_O_NET.jld2")
        θ_O_NET = JLD2.load(path, "params_experiment_idho_O_NET")
        path = joinpath(@__DIR__, "models", "experiment_idho_O_NET.jld2")
        O_NET = JLD2.load(path, "experiment_idho_O_NET")
        st_O_NET = JLD2.load(path, "st")
    end

    begin
        using JLD2, Lux
        path = joinpath(@__DIR__, "parameters", "params_experiment_idho_structured_ODE_NN.jld2")
        θ_Structured_ODE_NN = JLD2.load(path, "params_experiment_idho_structured_ODE_NN")
        path = joinpath(@__DIR__, "models", "experiment_idho_structured_ODE_NN.jld2")
        Structured_ODE_NN = JLD2.load(path, "experiment_idho_structured_ODE_NN")
        st_Structured_ODE_NN = JLD2.load(path, "st")
    end
end


# Generate ODE data of an isothermal damped harmonic oscillator
begin
    include("helpers/data_helper.jl")
    using Main.DataHelper: ODEfunc_idho, ODESolver
    # mass m, spring compliance c, the damping coefficient d and the environment temperature θ_0
    params = [2, 1, 0.5, 300]
    # initial state
    initial_state = [1.0, 1.0, 0.2]
    # Generate data set
    time_span = (0.0, 99.9)
    time_step = range(0.0, 99.9, 1000)
    ode_data = ODESolver(ODEfunc_idho, params, initial_state, time_span, time_step)
end


# Generate predict data
begin
    # Generate O-NET data
    using Main.DataHelper: NeuralODESolver
    function NeuralODE_O_NET(dz, z, θ_O_NET, t)
        dz[1] = O_NET(z, θ_O_NET, st_O_NET)[1][1]
        dz[2] = O_NET(z, θ_O_NET, st_O_NET)[1][2]
        dz[3] = O_NET(z, θ_O_NET, st_O_NET)[1][3]
    end
    predict_data_O_NET = NeuralODESolver(NeuralODE_O_NET, θ_O_NET, initial_state, time_span, time_step)
  
    # Generate O-NET data
    m = params[1]
    c = params[2]
    function NeuralODE_Structured_ODE_NN(dz, z, θ_Structured_ODE_NN, t)
        q = z[1]
        p = z[2]
        s_e = z[3]
        v = p/m
        dz[1] = v
        dz[2] = -q/c + Structured_ODE_NN([v], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][1]
        dz[3] = Structured_ODE_NN([v^2], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][2]
    end
    predict_data_Structured_ODE_NN = NeuralODESolver(NeuralODE_Structured_ODE_NN, θ_Structured_ODE_NN, initial_state, time_span, time_step)
end


#########
# Plots #
#########


# Phase portrait
begin
    using Plots
    plot(xlabel="q", ylabel="p", xlims=(-2,3), ylims=(-2,3))
    plot!(ode_data[1,:], ode_data[2,:], lw=3, label="Ground Truth", linestyle=:solid)
    plot!(predict_data_O_NET[1,:], predict_data_O_NET[2,:], lw=3, label="O-NET", linestyle=:dot)
    plot!(predict_data_Structured_ODE_NN[1,:], predict_data_Structured_ODE_NN[2,:], lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "phase_portrait_idho_O_NET_and_structured_ODE_NN.pdf"))
end


# Prediction error
begin
    using Plots
    l2_error_O_NET = vec(sum((ode_data .- predict_data_O_NET).^2, dims=1))
    l2_error_Structured_ODE_NN = vec(sum((ode_data .- predict_data_Structured_ODE_NN).^2, dims=1))
    plot(xlabel="Time Step", ylabel="L2 Error", xlims=(0,100), ylims=(0,0.05))
    plot!(time_step, l2_error_O_NET, lw=3, label="O-NET")
    plot!(time_step, l2_error_Structured_ODE_NN, lw=3, label="Structured ODE NN")
    #Plots.pdf(joinpath(@__DIR__, "figures", "prediction_error_idho_O_NET_and_structured_ODE_NN.pdf"))
end


# Hamiltonian evolution of an isothermal damped harmonic oscillator
begin
    using Plots
    Hamiltonian_Ground_Truth = ode_data[2,:].^2/(2*params[1]) + ode_data[1,:].^2/(2*params[2])
    Hamiltonian_O_NET = predict_data_O_NET[2,:].^2/(2*params[1]) + predict_data_O_NET[1,:].^2/(2*params[2])
    Hamiltonian_Structured_ODE_NN = predict_data_Structured_ODE_NN[2,:].^2/(2*params[1]) + predict_data_Structured_ODE_NN[1,:].^2/(2*params[2])
    plot(xlabel="Time Step", ylabel="Mechanical Energy", xlims=(0,100), ylims=(0.0,0.90))
    plot!(time_step, round.(Hamiltonian_Ground_Truth, digits=10), lw=2, label="Ground Truth", linestyle=:solid)
    plot!(time_step, round.(Hamiltonian_O_NET, digits=10), lw=2, label="O-NET", linestyle=:dot)
    plot!(time_step, round.(Hamiltonian_Structured_ODE_NN, digits=10), lw=2, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "Hamiltonian_evolution_idho_O_NET_and_structured_ODE_NN.pdf"))
end
