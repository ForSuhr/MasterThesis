begin
    include("helpers/data_helper.jl")
    include("helpers/train_helper.jl")
end


# Construct two neural networks for damping model, two for the thermal conduction model
begin
    using Lux, NNlib
    NN_d = Lux.Chain(Lux.Dense(3, 20, tanh),
                     Lux.Dense(20, 10),
                     Lux.Dense(10, 2))
    using Random
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    θ_NN_d, st_NN_d = Lux.setup(rng, NN_d)
end

begin
    using Lux, NNlib
    NN_tc = Lux.Chain(Lux.Dense(3, 20, tanh),
                      Lux.Dense(20, 10),
                      Lux.Dense(10, 2))
    using Random
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    θ_NN_tc, st_NN_tc = Lux.setup(rng, NN_tc)
end

# Combine the parameters of two neural networks
begin
    using Lux: ComponentArray
    θ_NN_d = Lux.ComponentArray(θ_NN_d)
    θ_NN_tc = Lux.ComponentArray(θ_NN_tc)
    θ_NN = Lux.ComponentArray{eltype(θ_NN_d)}()
    θ_NN = Lux.ComponentArray(θ_NN;θ_NN_d)
    θ_NN = Lux.ComponentArray(θ_NN;θ_NN_tc)

    # This is a combination of the parameters of four neural networks 
    θ_NN
end

# Generate ODE data of an non-isothermal damped harmonic oscillator
begin
    using Main.DataHelper: ODEfunc_ndho, ODESolver
    # mass m, spring compliance c, damping coefficient d, environment temperature θ_0, heat transfer coefficient α, heat capacity c_tc
    params = [2, 1, 0.5, 300, 0.2, 1.0]
    # initial state q, p, s_e, s_d
    initial_state = [1.0, 1.0, 0.2, 5.8]
    # Generate data set
    time_span = (0.0, 9.9)
    time_step = range(0.0, 9.9, 100)
    ode_data = ODESolver(ODEfunc_ndho, params, initial_state, time_span, time_step)
end


# Generate predict data
begin
    using Main.DataHelper: NeuralODESolver
    function NeuralODE(dz, z, θ_NN, t)
        q, p, s_e, s_d = z
        m, c, d, θ_0, α, c_tc = params
        v = p/m
        θ_d = exp(s_d/c_tc) / c_tc
        Δθ = θ_d - θ_0
        θ_NN_d = θ_NN.θ_NN_d
        θ_NN_tc = θ_NN.θ_NN_tc 
        # Compute the flow variable of damping
        output_NN_d = Array(transpose(NN_d([θ_0, v, Δθ], θ_NN_d, st_NN_d)[1]))
        transpose_output_NN_d = Array(transpose(output_NN_d))
        flow_d = 1/θ_0 .* transpose_output_NN_d * output_NN_d * [v, Δθ]
        # Compute the flow variable of thermal conduction
        output_NN_tc = Array(transpose(NN_tc([θ_0, Δθ, 0], θ_NN_tc, st_NN_tc)[1]))
        transpose_output_NN_tc = Array(transpose(output_NN_tc))
        flow_tc = 1/θ_0 .* transpose_output_NN_tc * output_NN_tc * [Δθ, 0]
        # System of ODEs
        dz[1] = v
        dz[2] = - q/c - flow_d[1]
        dz[3] = - flow_tc[1]
        dz[4] = - flow_d[2] - flow_tc[2]
    end
    using OrdinaryDiffEq: Tsit5
    predict_data = NeuralODESolver(NeuralODE, θ_NN, initial_state, time_span, time_step, Tsit5(), false)
end



############
# Training #
############

# Define physics priors
begin
    using LinearAlgebra: isposdef

    function DissipationMatrix(θ_NN, data, params)
        m, c, d, θ_0, α, c_tc = params
        M = Array{Float64, 3}(undef, (2,2,length(data[4,:])))
        for idx in range(1,length(data[4,:]))
            s_d = data[4, idx]
            θ_d = exp(s_d / c_tc) / c_tc
            Δθ = θ_d .- θ_0
            θ_NN_tc = θ_NN.θ_NN_tc
            output_NN_tc = Array(transpose(NN_tc([θ_0, Δθ, 0], θ_NN_tc, st_NN_tc)[1]))
            transpose_output_NN_tc = Array(transpose(output_NN_tc))
            M[:,:,idx] = transpose_output_NN_tc * output_NN_tc
        end
        return M
    end

    function TriangularMatrix(θ_NN, data, params)
        m, c, d, θ_0, α, c_tc = params
        D = Array{Float64, 2}(undef, (2,length(data[4,:])))
        for idx in range(1,length(data[4,:]))
            s_d = data[4, idx]
            θ_d = exp(s_d / c_tc) / c_tc
            θ_NN_tc = θ_NN.θ_NN_tc
            # When θ_0=0, D(θ_0,e)*e=0. This implies the first law
            output_NN_tc = Array(transpose(NN_tc([0, θ_d, 0], θ_NN_tc, st_NN_tc)[1]))
            D[:,idx] = output_NN_tc
        end
        return D
    end

    function FirstLawPrior(θ_NN, data, params)
        s_d = data[4,:]
        m, c, d, θ_0, α, c_tc = params
        θ_d = exp.(s_d ./ c_tc) / c_tc
        Δθ = θ_d .- θ_0
        D = TriangularMatrix(θ_NN, data, params)
        e = Array(transpose(hcat(θ_d, fill(θ_0, length(θ_d)))))
        first_law_prior = Array{Float64}(undef, length(data[4,:]))
        for idx in range(1, length(data[4,:]))
            first_law_prior[idx] = (Array(transpose(D[:,idx])) * e[:,idx] * Δθ[idx])[1]
        end
        return abs.(first_law_prior)
    end

    function SecondLawPrior(θ_NN, data, params, constant)
        M = DissipationMatrix(θ_NN, data, params)
        second_law_prior = Array{Float64}(undef, length(data[1,:]))
        for idx in range(1, length(data[1,:]))
            if isposdef(M[:,:,idx])
                second_law_prior[idx] = 0
            else
                second_law_prior[idx] = constant
            end
        end
        return second_law_prior
    end

    function RdSdF(θ_NN, data, params)
        m, c, d, θ_0, α, c_tc = params
        R_d__s_d__f =  Vector{Float64}(undef, length(data[2,:]))
        for idx in range(1,length(R_d__s_d__f))
            p = data[2, idx]
            s_d = data[4, idx]
            θ_d = exp(s_d / c_tc) / c_tc
            Δθ = θ_d - θ_0
            v = p/m
            θ_NN_d = θ_NN.θ_NN_d
            # Compute flow variables of damping
            output_NN_d = Array(transpose(NN_d([θ_0, v, Δθ], θ_NN_d, st_NN_d)[1]))
            transpose_output_NN_d = Array(transpose(output_NN_d))
            flow_d = 1/θ_0 .* transpose_output_NN_d * output_NN_d * [v, Δθ]
            f_θ_d__s_d__f[idx] = flow_d[2]
        end
        return R_d__s_d__f
    end

    function ReverseStepFunction(var, constant)
        return (var .> 0) .* constant
    end
end


# Construct loss function, callback function, dataloader and optimization function
begin
    using Main.TrainInterface: SolveIVP, OptFunction
    using Flux: DataLoader
    using OrdinaryDiffEq: Tsit5

    function loss_function(θ_NN, batch_data, batch_timesteps)
        pred_data = SolveIVP(NeuralODE, θ_NN, initial_state, batch_timesteps, Tsit5(), false)
        loss = sum( (batch_data[1,:] .- pred_data[1,:]) .^ 2 +
                    (batch_data[2,:] .- pred_data[2,:]) .^ 2 +
                    (batch_data[3,:] .- pred_data[3,:]) .^ 2 +
                    (batch_data[4,:] .- pred_data[4,:]) .^ 2)
        return loss, pred_data
    end

    callback = function(θ, loss, pred_data)
        println("loss: ", loss)
        return false
    end

    time_steps_1 = range(0.0, 0.9, 10)
    time_steps_2 = range(0.0, 2.9, 30)
    time_steps_3 = range(0.0, 4.9, 50)
    dataloader1 = DataLoader((ode_data[:,1:10], time_steps_1), batchsize = 10)
    dataloader2 = DataLoader((ode_data[:,1:30], time_steps_2), batchsize = 30)
    dataloader3 = DataLoader((ode_data[:,1:50], time_steps_3), batchsize = 50)

    using Optimization
    optf = OptFunction(loss_function, Optimization.AutoZygote())
    # optf = OptFunction(loss_function, Optimization.AutoReverseDiff())
    # optf = OptFunction(loss_function, Optimization.AutoFiniteDiff())
    # optf = OptFunction(loss_function, Optimization.AutoTracker())
    # optf = OptFunction(loss_function, Optimization.AutoForwardDiff())
end

# Training without physics priors
begin
    using Main.TrainInterface: LuxTrain
    α_learn = 0.001
    epochs = 200
    θ_NN = LuxTrain(optf, θ_NN, α_learn, epochs, dataloader3, callback)
end


begin
    using Main.TrainInterface: SolveIVP, OptFunction
    using Flux: DataLoader
    using OrdinaryDiffEq: Tsit5

    function loss_function(θ_NN, batch_data, batch_timesteps)
        pred_data = SolveIVP(NeuralODE, θ_NN, initial_state, batch_timesteps, Tsit5(), false)
        # First law prior
        first_law_prior = FirstLawPrior(θ_NN, batch_data, params)
        # Second law prior
        constant = 1
        second_law_prior = SecondLawPrior(θ_NN, batch_data, params, constant)
        # RdSdF should be negative
        # negative_prior = ReverseStepFunction(RdSdF(θ_NN, batch_data, params), constant)
        loss = sum( (batch_data[1,:] .- pred_data[1,:]) .^ 2 +
                    (batch_data[2,:] .- pred_data[2,:]) .^ 2 +
                    (batch_data[3,:] .- pred_data[3,:]) .^ 2 +
                    (batch_data[4,:] .- pred_data[4,:]) .^ 2 +
                    0.0001 * first_law_prior +
                    second_law_prior)
        return loss, pred_data
    end

    callback = function(θ, loss, pred_data)
        println("loss: ", loss)
        return false
    end

    time_steps_1 = range(0.0, 0.9, 10)
    time_steps_2 = range(0.0, 2.9, 30)
    time_steps_3 = range(0.0, 4.9, 50)
    dataloader1 = DataLoader((ode_data[:,1:10], time_steps_1), batchsize = 10)
    dataloader2 = DataLoader((ode_data[:,1:30], time_steps_2), batchsize = 30)
    dataloader3 = DataLoader((ode_data[:,1:50], time_steps_3), batchsize = 50)

    using Optimization
    # optf = OptFunction(loss_function, Optimization.AutoZygote())
    # optf = OptFunction(loss_function, Optimization.AutoReverseDiff())
    optf = OptFunction(loss_function, Optimization.AutoFiniteDiff())
    # optf = OptFunction(loss_function, Optimization.AutoTracker())
    # optf = OptFunction(loss_function, Optimization.AutoForwardDiff())
end


# Retrain with physics priors
begin
    using Main.TrainInterface: LuxTrain
    α_learn = 0.001
    epochs = 100
    θ_NN = LuxTrain(optf, θ_NN, α_learn, epochs, dataloader1, callback)
end

begin
    using Main.TrainInterface: LuxTrain
    α_learn = 0.001
    epochs = 100
    θ_NN = LuxTrain(optf, θ_NN, α_learn, epochs, dataloader2, callback)
end

begin
    using Main.TrainInterface: LuxTrain
    α_learn = 0.001
    epochs = 100
    θ_NN = LuxTrain(optf, θ_NN, α_learn, epochs, dataloader3, callback)
end


# Save the parameters and models of Structured ODE Neural Network
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "parameters", "params_ndho.jld2")
    JLD2.save(path, "params", θ_NN)
    path = joinpath(@__DIR__, "models", "model_ndho.jld2")
    JLD2.save(path, "model_d", NN_d,
                    "model_tc", NN_tc, 
                    "st_d", st_NN_d, 
                    "st_tc", st_NN_tc)
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
    path = joinpath(@__DIR__, "parameters", "params_ndho.jld2")
    θ_NN = JLD2.load(path, "params")
    path = joinpath(@__DIR__, "models", "model_ndho.jld2")
    NN_d = JLD2.load(path, "model_d")
    NN_tc = JLD2.load(path, "model_tc")
    st_NN_d = JLD2.load(path, "st_d")
    st_NN_tc = JLD2.load(path, "st_tc")
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
    function NeuralODE(dz, z, θ_NN, t)
        q, p, s_e, s_d = z
        m, c, d, θ_0, α, c_tc = params
        v = p/m
        θ_d = exp(s_d/c_tc) / c_tc
        Δθ = θ_d - θ_0
        θ_NN_d = θ_NN.θ_NN_d
        θ_NN_tc = θ_NN.θ_NN_tc 
        # Compute flow variables of damping
        output_NN_d = Array(transpose(NN_d([θ_0, v, Δθ], θ_NN_d, st_NN_d)[1]))
        transpose_output_NN_d = Array(transpose(output_NN_d))
        flow_d = 1/θ_0 .* transpose_output_NN_d * output_NN_d * [v, Δθ]
        # Compute flow variables of thermal conduction
        output_NN_tc = Array(transpose(NN_tc([θ_0, Δθ, 0], θ_NN_tc, st_NN_tc)[1]))
        transpose_output_NN_tc = Array(transpose(output_NN_tc))
        flow_tc = 1/θ_0 .* transpose_output_NN_tc * output_NN_tc * [Δθ, 0]
        # System of ODEs
        dz[1] = v
        dz[2] = - q/c - flow_d[1]
        dz[3] = - flow_tc[1]
        dz[4] = - flow_d[2] - flow_tc[2]
    end
    using OrdinaryDiffEq: Tsit5
    predict_data = NeuralODESolver(NeuralODE, θ_NN, initial_state, time_span, time_step, Tsit5(), false)
end

#########
# Plots #
#########


# Phase portrait
begin
    using Plots
    plot(xlabel="q", ylabel="p", xlims=(-2,3), ylims=(-2,3))
    plot!(ode_data[1,:], ode_data[2,:], lw=3, label="Ground Truth", linestyle=:solid)
    plot!(predict_data[1,:], predict_data[2,:], lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "phase_portrait_compositional_ndho.pdf"))
end


# Prediction error
begin
    using Plots
    l2_error_Structured_ODE_NN = vec(sum((ode_data .- predict_data).^2, dims=1))
    plot(xlabel="Time Step", ylabel="L2 Error")
    plot!(time_step, l2_error_Structured_ODE_NN, lw=3, label="Structured ODE NN")
    # Plots.pdf(joinpath(@__DIR__, "figures", "prediction_error_compositional_ndho.pdf"))
end


# Hamiltonian evolution of an isothermal damped harmonic oscillator
begin
    using Plots
    Hamiltonian_Ground_Truth = ode_data[2,:].^2/(2*params[1]) + ode_data[1,:].^2/(2*params[2])
    Hamiltonian_Structured_ODE_NN = predict_data[2,:].^2/(2*params[1]) + predict_data[1,:].^2/(2*params[2])
    plot(xlabel="Time Step", ylabel="Mechanical Energy", xlims=(0,100), ylims=(0.0,0.90))
    plot!(time_step, round.(Hamiltonian_Ground_Truth, digits=10), lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, round.(Hamiltonian_Structured_ODE_NN, digits=10), lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "Hamiltonian_evolution_compositional_ndho.pdf"))
end


# Compute effort and flow variables
begin
    p = ode_data[2,:]
    s_d = ode_data[4,:]
    m, c, d, θ_0, α, c_tc = params
    # Split the parameter combination into two components for the two neural networks
    θ_NN_d = θ_NN.θ_NN_d
    θ_NN_tc = θ_NN.θ_NN_tc      
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
        # Compute flow variables of damping
        output_NN_d = Array(transpose(NN_d([θ_0, v, Δθ], θ_NN_d, st_NN_d)[1]))
        transpose_output_NN_d = Array(transpose(output_NN_d))
        flow_d = 1/θ_0 .* transpose_output_NN_d * output_NN_d * [v, Δθ]
        # Compute flow variables of thermal conduction
        output_NN_tc = Array(transpose(NN_tc([θ_0, Δθ, 0], θ_NN_tc, st_NN_tc)[1]))
        transpose_output_NN_tc = Array(transpose(output_NN_tc))
        flow_tc = 1/θ_0 .* transpose_output_NN_tc * output_NN_tc * [Δθ, 0]
        # flow variables and dissipative power
        f_θ_d__p__f[idx] = flow_d[1]
        f_θ_d__s_d__f[idx] = flow_d[2]
        f_θ_tc__s_d__f[idx] = flow_tc[1]
        f_θ_tc__s_e__f[idx] = flow_tc[2]
        estimate_dissipative_power_d[idx] = v * f_θ_d__p__f[idx] + Δθ * f_θ_d__s_d__f[idx]
        estimate_dissipative_power_tc[idx] = Δθ * f_θ_tc__s_d__f[idx]
    end
end


# Plot the evolution of the flow variables
begin
    using Plots
    plot(xlabel="Time Step", ylabel="p.f")
    plot!(time_step, R_d__p__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_d__p__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "p.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="s_d.f of the damping")
    plot!(time_step, R_d__s_d__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_d__s_d__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "sd_d.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="s_d.f of the thermal conduction")
    plot!(time_step, R_tc__s_d__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_tc__s_d__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "sd_tc.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="sₑ.f")
    plot!(time_step, R_tc__s_e__f, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, f_θ_tc__s_e__f, lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "se.f_compositional_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="Dissipative Power of the damping")
    plot!(time_step, target_dissipative_power_d, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, estimate_dissipative_power_d, lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "dissipative_power_damping_ndho.pdf"))
end

begin
    using Plots
    plot(xlabel="Time Step", ylabel="Dissipative Power of the thermal conduction")
    plot!(time_step, target_dissipative_power_tc, lw=3, label="Ground Truth", linestyle=:solid)
    plot!(time_step, estimate_dissipative_power_tc, lw=3, label="Structured ODE NN", linestyle=:dash)
    # Plots.pdf(joinpath(@__DIR__, "figures", "dissipative_power_thermal_conduction_ndho.pdf"))
end
