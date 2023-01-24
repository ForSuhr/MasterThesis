begin
    include("helpers/data_helper.jl")
    include("helpers/train_helper.jl")
end


# Construct a Structured ODE Neural Network for damping
begin
    using Lux, NNlib
    Structured_ODE_NN_d = Lux.Chain(Lux.Dense(1, 20, tanh),
                                    Lux.Dense(20, 10),
                                    Lux.Dense(10, 2))
    using Random
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    θ_Structured_ODE_NN_d, st_Structured_ODE_NN_d = Lux.setup(rng, Structured_ODE_NN_d)
end


# Construct a Structured ODE Neural Network for thermal conduction
begin
    using Lux, NNlib
    Structured_ODE_NN_tc = Lux.Chain(Lux.Dense(1, 20, tanh),
                                     Lux.Dense(20, 10),
                                     Lux.Dense(10, 2))
    using Random
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    θ_Structured_ODE_NN_tc, st_Structured_ODE_NN_tc = Lux.setup(rng, Structured_ODE_NN_tc)
end


# Combine the parameters of two neural networks
begin
    using Lux: ComponentArray
    θ_Structured_ODE_NN_d = Lux.ComponentArray(θ_Structured_ODE_NN_d)
    θ_Structured_ODE_NN_tc = Lux.ComponentArray(θ_Structured_ODE_NN_tc)
    θ_Structured_ODE_NN = Lux.ComponentArray{eltype(θ_Structured_ODE_NN_d)}()
    θ_Structured_ODE_NN = Lux.ComponentArray(θ_Structured_ODE_NN;θ_Structured_ODE_NN_d)
    θ_Structured_ODE_NN = Lux.ComponentArray(θ_Structured_ODE_NN;θ_Structured_ODE_NN_tc)
    # This is a combination of the parameters of two neural networks 
    θ_Structured_ODE_NN  
end


# Generate ODE data of an non-isothermal damped harmonic oscillator
begin
    using Main.DataHelper: ODEfunc_ndho, ODESolver
    # mass m, spring compliance c, damping coefficient d, environment temperature θ_0, heat transfer coefficient α, heat capacity c_tc
    params = [2, 1, 0.5, 300, 0.2, 1.0]
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
    function NeuralODE_Structured_ODE_NN(dz, z, θ_NN, t)
        q, p, s_e, s_d = z
        m, c, d, θ_0, α, c_tc = params
        v = p/m
        θ_d = exp(s_d/c_tc) / c_tc
        Δθ = θ_d - θ_0
        θ_NN_d = θ_NN.θ_Structured_ODE_NN_d
        θ_NN_tc = θ_NN.θ_Structured_ODE_NN_tc       
        dz[1] = v
        dz[2] = - q/c - Structured_ODE_NN_d([v], θ_NN_d, st_Structured_ODE_NN_d)[1][1]
        dz[3] = - Structured_ODE_NN_tc([Δθ/θ_0], θ_NN_tc, st_Structured_ODE_NN_tc)[1][2]
        dz[4] = - Structured_ODE_NN_d([-(v^2)/θ_d], θ_NN_d, st_Structured_ODE_NN_d)[1][2] - Structured_ODE_NN_tc([Δθ/θ_d], θ_NN_tc, st_Structured_ODE_NN_tc)[1][1]
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
        loss = sum(  (batch_data[1,:] .- pred_data[1,:]) .^ 2 +
                     (batch_data[2,:] .- pred_data[2,:]) .^ 2 +
                     (batch_data[3,:] .- pred_data[3,:]) .^ 2 +
                     (batch_data[4,:] .- pred_data[4,:]) .^ 2)
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
    α_learn = 0.0001
    epochs = 1000
    θ_Structured_ODE_NN = LuxTrain(optf_Structured_ODE_NN, θ_Structured_ODE_NN, α_learn, epochs, dataloader, callback_Structured_ODE_NN)
end


# Save the parameters and models of Structured ODE Neural Network
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "parameters", "params_compositional_experiment_ndho.jld2")
    JLD2.save(path, "params", θ_Structured_ODE_NN)
    path = joinpath(@__DIR__, "models", "compositional_experiment_ndho.jld2")
    JLD2.save(path, "model_d", Structured_ODE_NN_d,
                    "model_tc", Structured_ODE_NN_tc, 
                    "st_d", st_Structured_ODE_NN_d, 
                    "st_tc", st_Structured_ODE_NN_tc)
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
    θ_Structured_ODE_NN = JLD2.load(path, "params")
    path = joinpath(@__DIR__, "models", "compositional_experiment_ndho.jld2")
    Structured_ODE_NN_d = JLD2.load(path, "model_d")
    Structured_ODE_NN_tc = JLD2.load(path, "model_tc")
    st_Structured_ODE_NN_d = JLD2.load(path, "st_d")
    st_Structured_ODE_NN_tc = JLD2.load(path, "st_tc")
end

# Generate ODE data of an non-isothermal damped harmonic oscillator
begin
    using Main.DataHelper: ODEfunc_ndho, ODESolver
    # mass m, spring compliance c, damping coefficient d, environment temperature θ_0, heat transfer coefficient α, heat capacity c_tc
    params = [2, 1, 0.5, 300, 0.2, 1.0]
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
    function NeuralODE_Structured_ODE_NN(dz, z, θ_NN, t)
        q, p, s_e, s_d = z
        m, c, d, θ_0, α, c_tc = params
        v = p/m
        θ_d = exp(s_d/c_tc) / c_tc
        Δθ = θ_d - θ_0
        θ_NN_d = θ_NN.θ_Structured_ODE_NN_d
        θ_NN_tc = θ_NN.θ_Structured_ODE_NN_tc       
        dz[1] = v
        dz[2] = - q/c - Structured_ODE_NN_d([v], θ_NN_d, st_Structured_ODE_NN_d)[1][1]
        dz[3] = - Structured_ODE_NN_tc([Δθ/θ_0], θ_NN_tc, st_Structured_ODE_NN_tc)[1][2]
        dz[4] = - Structured_ODE_NN_d([-(v^2)/θ_d], θ_NN_d, st_Structured_ODE_NN_d)[1][2] - Structured_ODE_NN_tc([Δθ/θ_d], θ_NN_tc, st_Structured_ODE_NN_tc)[1][1]
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
    plot!(predict_data_Structured_ODE_NN[1,:], predict_data_Structured_ODE_NN[2,:], lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "phase_portrait_compositional_ndho.pdf"))
end


# Prediction error
begin
    using Plots
    l2_error_Structured_ODE_NN = vec(sum((ode_data .- predict_data_Structured_ODE_NN).^2, dims=1))
    plot(xlabel="Time Step", ylabel="L2 Error", xlims=(0,100), ylims=(0,0.00005))
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
    m, c, d, θ_0, α, c_tc = params
    # Split the parameter combination into two components for the two neural networks
    θ_NN_d = θ_Structured_ODE_NN.θ_Structured_ODE_NN_d
    θ_NN_tc = θ_Structured_ODE_NN.θ_Structured_ODE_NN_tc      
    # Target values
    v = p ./ m
    θ_d = exp.(s_d ./ c_tc) / c_tc
    R_d__p__f = d * v
    R_d__s_d__f = - d * v.^2 ./ θ_d
    R_tc__s_d__f = α .* (θ_d .- θ_0) ./ θ_d
    R_tc__s_e__f = α .* (θ_0 .- θ_d) ./ θ_0
    target_dissipative_power_d = θ_0 * d * v .^2 ./ θ_d
    target_dissipative_power_tc = θ_0 * α * ((θ_d .- θ_0) ./ θ_0 .+ (θ_0 .- θ_d) ./ θ_d)
    # Estimated values
    f_θ_d__p__f = Vector{Float64}(undef, length(ode_data[2,:]))
    f_θ_d__s_d__f = Vector{Float64}(undef, length(ode_data[2,:]))
    f_θ_tc__s_d__f = Vector{Float64}(undef, length(ode_data[4,:]))
    f_θ_tc__s_e__f = Vector{Float64}(undef, length(ode_data[4,:]))
    estimate_dissipative_power_d = Vector{Float64}(undef, length(ode_data[2,:]))
    estimate_dissipative_power_tc = Vector{Float64}(undef, length(ode_data[4,:]))
    for idx in range(1,length(ode_data[2,:]))
        q = ode_data[1, idx]
        p = ode_data[2, idx]
        s_e = ode_data[3, idx]
        s_d = ode_data[4, idx]
        θ_d = exp(s_d / c_tc) / c_tc
        Δθ = θ_d - θ_0
        v = p/m
        f_θ_d__p__f[idx] = Structured_ODE_NN_d([v], θ_NN_d, st_Structured_ODE_NN_d)[1][1]
        f_θ_d__s_d__f[idx] = Structured_ODE_NN_d([-(v^2)/θ_d], θ_NN_d, st_Structured_ODE_NN_d)[1][2]
        f_θ_tc__s_d__f[idx] = Structured_ODE_NN_tc([Δθ/θ_d], θ_NN_tc, st_Structured_ODE_NN_tc)[1][1]
        f_θ_tc__s_e__f[idx] = Structured_ODE_NN_tc([Δθ/θ_0], θ_NN_tc, st_Structured_ODE_NN_tc)[1][2]
        estimate_dissipative_power_d[idx] = v * Structured_ODE_NN_d([v], θ_NN_d, st_Structured_ODE_NN_d)[1][1] + Δθ * Structured_ODE_NN_d([-(v^2)/θ_d], θ_NN_d, st_Structured_ODE_NN_d)[1][2]
        estimate_dissipative_power_tc[idx] = Δθ * Structured_ODE_NN_tc([Δθ/θ_d], θ_NN_tc, st_Structured_ODE_NN_tc)[1][1]
    end
end


# Plot the evolution of the flow variables
begin
    using Plots
    plot(xlabel="Time Step", ylabel="p.f", xlims=(0,100), ylims=(-0.5,0.4))
    plot!(time_step, R_d__p__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_d__p__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "p.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="s_d.f of the damping", xlims=(0,100), ylims=(-0.01,0.01))
    plot!(time_step, R_d__s_d__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_d__s_d__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "sd_d.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="s_d.f of the thermal conduction", xlims=(0,100), ylims=(-0.01,0.04))
    plot!(time_step, R_tc__s_d__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_tc__s_d__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "sd_tc.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="sₑ.f", xlims=(0,100), ylims=(-0.05,0.05))
    plot!(time_step, R_tc__s_e__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_tc__s_e__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "se.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="Dissipative Power of the damping", xlims=(0,100), ylims=(-0.2,0.4))
    plot!(time_step, target_dissipative_power_d, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, estimate_dissipative_power_d, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "dissipative_power_damping_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="Dissipative Power of the thermal conduction", xlims=(0,100), ylims=(-0.4,1.0))
    plot!(time_step, target_dissipative_power_tc, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, estimate_dissipative_power_tc, lw=3, label="Structured ODE NN", linestyle=:dash)
    #Plots.pdf(joinpath(@__DIR__, "figures", "dissipative_power_thermal_conduction_ndho.pdf"))
end
