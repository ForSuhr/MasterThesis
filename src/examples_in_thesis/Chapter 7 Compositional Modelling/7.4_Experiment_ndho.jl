begin
    include("helpers/data_helper.jl")
    include("helpers/train_helper.jl")
end


# Construct a Structured ODE Neural Network
begin
    using Lux, NNlib
    Structured_ODE_NN = Lux.Chain(Lux.Dense(1, 20, tanh),
                                  Lux.Dense(20, 10, tanh),
                                  Lux.Dense(10, 4))

    using Random
    rng = Random.default_rng()
    θ_Structured_ODE_NN, st_Structured_ODE_NN = Lux.setup(rng, Structured_ODE_NN)
end


# Generate ODE data of an non-isothermal damped harmonic oscillator
begin
    using Main.DataHelper: ODEfunc_ndho, ODESolver
    # mass m, spring compliance c, damping coefficient d, environment temperature θ_0, heat transfer coefficient α, thermal capacity parameters c₁ and c₂
    params = [2, 1, 0.5, 300, 0.2, 1.0, 1.0]
    # initial state q, p, s_e, s_d
    initial_state = [1.0, 1.0, 0.2, 5.8]
    # Generate data set
    time_span = (0.0, 19.9)
    time_step = range(0.0, 19.9, 200)
    ode_data = ODESolver(ODEfunc_ndho, params, initial_state, time_span, time_step)
end


# Generate predict data
begin
    using Main.DataHelper: NeuralODESolver
    function NeuralODE_Structured_ODE_NN(dz, z, θ_Structured_ODE_NN, t)
        q, p, s_e, s_d = z
        m, c, d, θ_0, α, c₁, c₂ = params
        v = p/m
        dz[1] = v
        dz[2] = - q/c - Structured_ODE_NN([v], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][1]
        dz[3] = - Structured_ODE_NN([exp(s_d)/θ_0], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][2]
        dz[4] = - Structured_ODE_NN([v^2/exp(s_d)], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][3] - Structured_ODE_NN([-θ_0/exp(s_d)], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][4]
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
    using OrdinaryDiffEq: Tsit5

    function loss_function_Structured_ODE_NN(θ, batch_data, batch_timesteps)
        pred_data = SolveIVP(NeuralODE_Structured_ODE_NN, θ, initial_state, batch_timesteps, Tsit5(), false)
        loss = sum((batch_data[1,:] .- pred_data[1,:]) .^ 2 +
                   (batch_data[2,:] .- pred_data[2,:]) .^ 2 +
                   (batch_data[3,:] .- pred_data[3,:]) .^ 2 +
                 10(batch_data[4,:] .- pred_data[4,:]) .^ 2)
        return loss, pred_data
    end

    callback_Structured_ODE_NN = function(θ, loss, pred_data)
        println(loss_function_Structured_ODE_NN(θ, ode_data, time_step)[1])
        return false
    end

    dataloader = DataLoader((ode_data, time_step), batchsize = 200)

    optf_Structured_ODE_NN = OptFunction(loss_function_Structured_ODE_NN)
end


# Repeat training for the Structured ODE Neural Network
begin
    using Main.TrainInterface: LuxTrain
    α_learn = 0.001
    epochs = 1000
    θ_Structured_ODE_NN = LuxTrain(optf_Structured_ODE_NN, θ_Structured_ODE_NN, α_learn, epochs, dataloader, callback_Structured_ODE_NN)
end


# Save the parameters and models of Structured ODE Neural Network
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "parameters", "params_compositional_experiment_ndho.jld2")
    JLD2.save(path, "params_compositional_experiment_ndho", θ_Structured_ODE_NN)
    path = joinpath(@__DIR__, "models", "compositional_experiment_ndho.jld2")
    JLD2.save(path, "compositional_experiment_ndho", Structured_ODE_NN, "st", st_Structured_ODE_NN)
end




##############
# Evaluation #
##############
begin
    include("helpers/data_helper.jl")
    include("helpers/train_helper.jl")
end

# Load the parameters and models
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "parameters", "params_compositional_experiment_ndho.jld2")
    θ_Structured_ODE_NN = JLD2.load(path, "params_compositional_experiment_ndho")
    path = joinpath(@__DIR__, "models", "compositional_experiment_ndho.jld2")
    Structured_ODE_NN = JLD2.load(path, "compositional_experiment_ndho")
    st_Structured_ODE_NN = JLD2.load(path, "st")
end

# Generate ODE data of an non-isothermal damped harmonic oscillator
begin
    using Main.DataHelper: ODEfunc_ndho, ODESolver
    # mass m, spring compliance c, damping coefficient d, environment temperature θ_0, heat transfer coefficient α, thermal capacity parameters c₁ and c₂
    params = [2, 1, 0.5, 300, 0.2, 1.0, 1.0]
    # initial state q, p, s_e, s_d
    initial_state = [1.0, 1.0, 0.2, 5.8]
    # Generate data set
    time_span = (0.0, 99.9)
    time_step = range(0.0, 99.9, 1000)
    ode_data = ODESolver(ODEfunc_ndho, params, initial_state, time_span, time_step)
end


# Generate predict data
begin
    using Main.DataHelper: NeuralODESolver
    function NeuralODE_Structured_ODE_NN(dz, z, θ_Structured_ODE_NN, t)
        q, p, s_e, s_d = z
        m, c, d, θ_0, α, c₁, c₂ = params
        v = p/m
        dz[1] = v
        dz[2] = - q/c - Structured_ODE_NN([v], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][1]
        dz[3] = - Structured_ODE_NN([exp(s_d)/θ_0], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][2]
        dz[4] = - Structured_ODE_NN([v^2/exp(s_d)], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][3] - Structured_ODE_NN([-θ_0/exp(s_d)], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][4]
    end
    using OrdinaryDiffEq: Tsit5
    predict_data_Structured_ODE_NN = NeuralODESolver(NeuralODE_Structured_ODE_NN, θ_Structured_ODE_NN, initial_state, time_span, time_step, Tsit5(), false)
end

#########
# Plots #
#########


# Phase portrait
begin
    using Plots
    plot(xlabel="q", ylabel="p", xlims=(-2,3), ylims=(-2,3))
    plot!(ode_data[1,:], ode_data[2,:], lw=3, label="Ground Truth", linestyle=:solid)
    plot!(predict_data_Structured_ODE_NN[1,:], predict_data_Structured_ODE_NN[2,:], lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "phase_portrait_compositional_ndho.pdf"))
end


# Prediction error
begin
    using Plots
    l2_error_Structured_ODE_NN = vec(sum((ode_data .- predict_data_Structured_ODE_NN).^2, dims=1))
    plot(xlabel="Time Step", ylabel="L2 Error", xlims=(0,100), ylims=(0,0.02))
    plot!(time_step, l2_error_Structured_ODE_NN, lw=3, label="Structured ODE NN")
    #Plots.pdf(joinpath(@__DIR__, "figures", "prediction_error_compositional_ndho.pdf"))
end


# Hamiltonian evolution of an isothermal damped harmonic oscillator
begin
    using Plots
    Hamiltonian_Ground_Truth = ode_data[2,:].^2/(2*params[1]) + ode_data[1,:].^2/(2*params[2])
    Hamiltonian_Structured_ODE_NN = predict_data_Structured_ODE_NN[2,:].^2/(2*params[1]) + predict_data_Structured_ODE_NN[1,:].^2/(2*params[2])
    plot(xlabel="Time Step", ylabel="Mechanical Energy", xlims=(0,100), ylims=(0.0,0.90))
    plot!(time_step, round.(Hamiltonian_Ground_Truth, digits=10), lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, round.(Hamiltonian_Structured_ODE_NN, digits=10), lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "Hamiltonian_evolution_compositional_ndho.pdf"))
end


# Compute effort and flow variables
begin
    p = ode_data[2,:]
    s_d = ode_data[4,:]
    m, c, d, θ_0, α, c₁, c₂ = params
    # Target values
    R_d__p__f = d * p ./ m
    θ_d = c₁/c₂ * exp.(s_d ./ c₂)
    R_d__s_e__f = α .* (θ_0 .- θ_d) ./ θ_0
    # Estimated values
    f_θ__p__f = Vector{Float64}(undef, length(ode_data[2,:]))
    f_θ__s_e__f = Vector{Float64}(undef, length(ode_data[2,:]))
    for (idx, p) in enumerate(ode_data[2,:])
        v = p/m
        f_θ__p__f[idx] = Structured_ODE_NN([v], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][1]
    end
    for (idx, s_d) in enumerate(ode_data[4,:])
        f_θ__s_e__f[idx] = Structured_ODE_NN([exp(s_d)/θ_0], θ_Structured_ODE_NN, st_Structured_ODE_NN)[1][2]
    end
end


# Plot the evolution of the flow variables

begin
    using Plots
    plot(xlabel="Time Step", ylabel="p.f", xlims=(0,100), ylims=(-0.5,0.4))
    plot!(time_step, R_d__p__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ__p__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "p.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="sₑ.f", xlims=(0,100), ylims=(-0.06,0.03))
    plot!(time_step, R_d__s_e__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ__s_e__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "se.f_compositional_ndho.pdf"))
end
