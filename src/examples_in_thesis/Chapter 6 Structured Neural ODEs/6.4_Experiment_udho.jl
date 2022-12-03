# Load O_NET
begin
    using Lux, Flux, JLD2
    # Load the parameters
    begin
        path = joinpath(@__DIR__, "parameters", "params_O_NET.jld2")
        θ_O_NET = JLD2.load(path, "params_O_NET")
    end
    # Load the model
    begin
        path = joinpath(@__DIR__, "models", "O_NET.jld2")
        O_NET = JLD2.load(path, "O_NET")
        re_O_NET = JLD2.load(path, "re")
    end
end


# Load H-NET
begin
    using Lux, Flux, JLD2
    # Load the parameters
    begin
        path = joinpath(@__DIR__, "parameters", "params_H_NET.jld2")
        θ_H_NET = JLD2.load(path, "params_H_NET")
    end
    # Load the model
    begin
        path = joinpath(@__DIR__, "models", "H_NET.jld2")
        H_NET = JLD2.load(path, "H_NET")
        st_H_NET = JLD2.load(path, "st")
    end
end


# Load HNN
begin
    using Lux, Flux, JLD2
    # Load the parameters
    begin
        path = joinpath(@__DIR__, "parameters", "params_HNN.jld2")
        θ_HNN = JLD2.load(path, "params_HNN")
    end
    # Load the model
    begin
        path = joinpath(@__DIR__, "models", "HNN.jld2")
        HNN = JLD2.load(path, "HNN")
        st_HNN = JLD2.load(path, "st")
    end
end


# Load Structured ODE Neural Network
begin
    using Lux, Flux, JLD2
    # Load the parameters
    begin
        path = joinpath(@__DIR__, "parameters", "params_structured_ODE_NN.jld2")
        θ_structured_ODE_NN = JLD2.load(path, "params_structured_ODE_NN")
    end
    # Load the model
    begin
        path = joinpath(@__DIR__, "models", "structured_ODE_NN.jld2")
        Structured_ODE_NN = JLD2.load(path, "structured_ODE_NN")
        st_structured_ODE_NN = JLD2.load(path, "st")
    end
end


# Generate ODE data
begin
    include("helpers/data_helper.jl")
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
    # Generate O-NET data
    using Main.DataHelper: NeuralODESolver, SymplecticGradient
    function NeuralODE_O_NET(dz, z, θ, t)
        dz[1] = re_O_NET(θ_O_NET)(z)[1]
        dz[2] = re_O_NET(θ_O_NET)(z)[2]
    end
    predict_data_O_NET = NeuralODESolver(NeuralODE_O_NET, θ_O_NET, initial_state, time_span, time_step)

    # Generate H-NET data
    function NeuralODE_H_NET(z, θ, t)
        dz = SymplecticGradient(H_NET, θ_H_NET, st_H_NET, z)
    end
    predict_data_H_NET = NeuralODESolver(NeuralODE_H_NET, θ_H_NET, initial_state, time_span, time_step)

    # Generate HNN data
    function NeuralODE_HNN(z, θ, t)
        dz = SymplecticGradient(HNN, θ_HNN, st_HNN, z)
    end
    predict_data_HNN = NeuralODESolver(NeuralODE_HNN, θ_HNN, initial_state, time_span, time_step)
    
    # Generate O-NET data
    m = params[1]
    function NeuralODE_Structured_ODE_NN(dz, z, θ, t)
        q = z[1]
        p = z[2]
        dz[1] = p/m
        dz[2] = Structured_ODE_NN([q], θ_structured_ODE_NN, st_structured_ODE_NN)[1][1]
    end
    predict_data_Structured_ODE_NN = NeuralODESolver(NeuralODE_Structured_ODE_NN, θ_structured_ODE_NN, initial_state, time_span, time_step)
end


# Plots

# Phase portrait
begin
    using Plots
    plot(xlabel="q", ylabel="p", xlims=(-2,3), ylims=(-2,3), fmt=:PDF)
    plot!(ode_data[1,:], ode_data[2,:], lw=3, label="Ground Truth", linestyle=:solid)
    plot!(predict_data_H_NET[1,:], predict_data_H_NET[2,:], lw=3, label="H-NET", linestyle=:dot)
    plot!(predict_data_HNN[1,:], predict_data_HNN[2,:], lw=3, label="HNN", linestyle=:dash)
    Plots.pdf(joinpath(@__DIR__, "figures", "phase_portrait_H_NET_and_HNN.pdf"))
end


# Phase portrait
begin
    using Plots
    plot(xlabel="q", ylabel="p", xlims=(-2,3), ylims=(-2,3))
    plot!(ode_data[1,:], ode_data[2,:], lw=3, label="Ground Truth", linestyle=:solid)
    plot!(predict_data_O_NET[1,:], predict_data_O_NET[2,:], lw=3, label="O-NET", linestyle=:dot)
    plot!(predict_data_Structured_ODE_NN[1,:], predict_data_Structured_ODE_NN[2,:], lw=3, label="Structured ODE NN", linestyle=:dash)
    Plots.pdf(joinpath(@__DIR__, "figures", "phase_portrait_O_NET_and_structured_ODE_NN.pdf"))
end


# Prediction error
begin
    using Plots
    l2_error_H_NET = vec(sum((ode_data .- predict_data_H_NET).^2, dims=1))
    l2_error_HNN = vec(sum((ode_data .- predict_data_HNN).^2, dims=1))
    plot(xlabel="Time Step", ylabel="L2 Error", xlims=(0,10), ylims=(0,0.005))
    plot!(time_step, l2_error_H_NET, lw=3, label="H-NET")
    plot!(time_step, l2_error_HNN, lw=3, label="HNN")
    Plots.pdf(joinpath(@__DIR__, "figures", "prediction_error_H_NET_and_HNN.pdf"))
end


# Prediction error
begin
    using Plots
    l2_error_O_NET = vec(sum((ode_data .- predict_data_O_NET).^2, dims=1))
    l2_error_Structured_ODE_NN = vec(sum((ode_data .- predict_data_Structured_ODE_NN).^2, dims=1))
    plot(xlabel="Time Step", ylabel="L2 Error", xlims=(0,10), ylims=(0,0.01))
    plot!(time_step, l2_error_O_NET, lw=3, label="O-NET")
    plot!(time_step, l2_error_Structured_ODE_NN, lw=3, label="Structured ODE NN")
    Plots.pdf(joinpath(@__DIR__, "figures", "prediction_error_O_NET_and_structured_ODE_NN.pdf"))
end


# Hamiltonian evolution of an undamped harmonic oscillator
begin
    using Plots
    Hamiltonian_Ground_Truth = ode_data[2,:].^2/(2*params[1]) + ode_data[1,:].^2/(2*params[2])
    Hamiltonian_H_NET = predict_data_H_NET[2,:].^2/(2*params[1]) + predict_data_H_NET[1,:].^2/(2*params[2])
    Hamiltonian_HNN = predict_data_HNN[2,:].^2/(2*params[1]) + predict_data_HNN[1,:].^2/(2*params[2])
    plot(xlabel="Time Step", ylabel="Hamiltonian", xlims=(0,10), ylims=(0.72,0.80))
    plot!(time_step, round.(Hamiltonian_Ground_Truth, digits=10), label="Ground Truth")
    plot!(time_step, round.(Hamiltonian_H_NET, digits=10), label="H-NET")
    plot!(time_step, round.(Hamiltonian_HNN, digits=10), label="HNN")
    Plots.pdf(joinpath(@__DIR__, "figures", "Hamiltonian_evolution_H_NET_and_HNN.pdf"))
end


# Hamiltonian evolution of an undamped harmonic oscillator
begin
    using Plots
    Hamiltonian_Ground_Truth = ode_data[2,:].^2/(2*params[1]) + ode_data[1,:].^2/(2*params[2])
    Hamiltonian_O_NET = predict_data_O_NET[2,:].^2/(2*params[1]) + predict_data_O_NET[1,:].^2/(2*params[2])
    Hamiltonian_Structured_ODE_NN = predict_data_Structured_ODE_NN[2,:].^2/(2*params[1]) + predict_data_Structured_ODE_NN[1,:].^2/(2*params[2])
    plot(xlabel="Time Step", ylabel="Hamiltonian", xlims=(0,10), ylims=(0.65,0.90))
    plot!(time_step, round.(Hamiltonian_Ground_Truth, digits=10), label="Ground Truth")
    plot!(time_step, round.(Hamiltonian_O_NET, digits=10), label="O-NET")
    plot!(time_step, round.(Hamiltonian_Structured_ODE_NN, digits=10), label="Structured ODE NN")
    Plots.pdf(joinpath(@__DIR__, "figures", "Hamiltonian_evolution_O_NET_and_structured_ODE_NN.pdf"))
end
