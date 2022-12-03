# This is an example for using saved models


# Load O_NET
begin
    using Lux, Flux, JLD2
    # Load the parameters
    begin
        path = joinpath(@__DIR__, "Chapter 6 Structured Neural ODEs/parameters", "params_O_NET.jld2")
        θ = JLD2.load(path, "params_O_NET")
    end
    # Load the model
    begin
        path = joinpath(@__DIR__, "Chapter 6 Structured Neural ODEs/models", "O_NET.jld2")
        O_NET = JLD2.load(path, "O_NET")
        re = JLD2.load(path, "re")
    end
end


# Generate ODE data
begin
    include("Chapter 6 Structured Neural ODEs/helpers/data_helper.jl")
    using Main.DataHelper: ODEfunc_udho, ODESolver
    # mass m and spring compliance c
    params = [2, 1]
    # initial state
    initial_state = [1.0, 1.0]
    # Generate data set
    time_span = (0.0, 9.9)
    time_step = range(0.0, 9.9, 100)
    ode_data = ODESolver(ODEfunc_udho, params, initial_state, time_span, time_step)
end


# Generate predict data
begin
    using Main.DataHelper: NeuralODESolver
    function NeuralODE(dz, z, θ, t)
        dz[1] = re(θ)(z)[1]
        dz[2] = re(θ)(z)[2]
    end
    predict_data = NeuralODESolver(NeuralODE, θ, initial_state, time_span, time_step)
end


# Plot trajectories
begin
    using Plots
    plot(ode_data[1,:], ode_data[2,:], lw=3, xlabel="q", ylabel="p", label="Ground Truth")
    plot!(predict_data[1,:], predict_data[2,:], lw=3, label="O-NET")
end