######################################
# Step 1: construct a neural network #
######################################

using Lux
Structured_O_NET = Lux.Chain(Lux.Dense(1, 20, tanh),
                             Lux.Dense(20, 10, tanh),
                             Lux.Dense(10, 2))

using Random
rng = Random.default_rng()
ps, st = Lux.setup(rng, Structured_O_NET)


############################
# Step 2: construct an IVP #
############################
m = 2
c = 1
θ_0 = 20
function ODE(dz, z, θ, t)
    q = z[1]
    p = z[2]
    v = p/m
    dz[1] = v
    dz[2] = -q/c + Structured_O_NET([v], θ, st)[1][1]
    dz[3] = Structured_O_NET([v^2], θ, st)[1][2] / θ_0
end

# initial state
initial_state = [1.0, 1.0, 0.5]
    
# Starting at 0.0 and ending at 19.9, the length of each step is 0.1. Thus, we have 200 time steps in total.
time_span = (0.0, 19.9)
time_steps = range(0.0, 19.9, 200)

# parameters of the neural network
θ = ps

# ODEProblem is an IVP constructor in the Julia package SciMLBase.jl
using SciMLBase
IVP = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, time_span, θ)




#########################
# Step 3: solve the IVP #
#########################

# Select a numerical method to solve the IVP
using OrdinaryDiffEq
numerical_method = ImplicitMidpoint()

# Select the adjoint method to computer the gradient of the loss with respect to the parameters. 
# ReverseDiffVJP is a callable function in the package SciMLSensitivity.jl, it uses the automatic 
# differentiation tool ReverseDiff.jl to compute the vector-Jacobian products (VJP) efficiently. 
using ReverseDiff
using SciMLSensitivity
sensitivity_analysis = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

# Use the ODE Solver CommonSolve.solve to yield solution. And the solution is the estimate of the coordinates trajectories.
using CommonSolve
solution = CommonSolve.solve(IVP, numerical_method, p=θ, tstops = time_steps, sensealg=sensitivity_analysis)
# Convert the solution into a 2D-array
pred_data = Array(solution)




#####################################
# Step 4: construct a loss function #
#####################################

function ODEfunc_idho(dz, z, params, t)
    q, p = z
    m, c, d, θ_0 = params
    v = p/m
    dz[1] = v
    dz[2] = -q/c -d*v
    dz[3] = d*v^2/θ_0
end

params = [2, 1, 0.5, 20] 
prob = ODEProblem(ODEFunction(ODEfunc_idho), initial_state, time_span, params)
ode_data = Array(CommonSolve.solve(prob, numerical_method, tstops = time_steps))
using Plots
plot(ode_data[1,:], ode_data[2,:])
plot!(pred_data[1,:], ode_data[2,:])

function solve_IVP(θ, batch_timesteps)
    IVP = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, (batch_timesteps[1], batch_timesteps[end]), θ)
    pred_data = Array(CommonSolve.solve(IVP, numerical_method, p=θ, tstops = batch_timesteps, sensealg=sensitivity_analysis))
    return pred_data
end

function loss_function(θ, batch_data, batch_timesteps)
    pred_data = solve_IVP(θ, batch_timesteps)
    # the entroy needs a larger weight for better training
    loss = sum((batch_data[1,:] .- pred_data[1,:]) .^ 2 +
               (batch_data[2,:] .- pred_data[2,:]) .^ 2 +
            100(batch_data[3,:] .- pred_data[3,:]) .^ 2)
    return loss, pred_data
end

callback = function(θ, loss, pred_data)
    println(loss_function(θ, ode_data, time_steps)[1])
    return false
end




####################################
# Step 5: train the neural network #
####################################

# The dataloader generates a batch of data according to the given batchsize from the "ode_data".
using Flux: DataLoader
dataloader = DataLoader((ode_data, time_steps), batchsize = 200)

# Select an automatic differentiation tool
using Optimization
adtype = Optimization.AutoZygote()
# Construct an optimization problem with the given automatic differentiation and the initial parameters θ
optf = Optimization.OptimizationFunction((θ, ps, batch_data, batch_timesteps) -> loss_function(θ, batch_data, batch_timesteps), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(θ))
# Train the model multiple times. The "ncycle" is a function in the package IterTools.jl, it cycles through the dataloader "epochs" times.
using OptimizationOptimisers
using IterTools: ncycle
epochs = 10;
result = Optimization.solve(optprob, Optimisers.ADAM(0.01), ncycle(dataloader, epochs), callback=callback)
# Access the trained parameters
θ = result.u

# The "loss_function" returns a tuple, where the first element of the tuple is the loss
loss = loss_function(result.u, ode_data, time_steps)[1]

# Option: continue the training
include("helpers/train_helper.jl")
using Main.TrainInterface: LuxTrain
# Adjust the learning rate and epochs, then repeat this code block
begin
    α = 0.0001
    epochs = 10
    θ = LuxTrain(optf, θ, α, epochs, dataloader, callback)
end




##########################
# Step 6: test the model #
##########################

# Plot phase portrait
trained_solution = CommonSolve.solve(IVP, numerical_method, p=θ, tstops = time_steps, sensealg=sensitivity_analysis)
plot(ode_data[1,:], ode_data[2,:], ode_data[3,:], label="Ground truth", lw=2, xlabel="q", ylabel="p", zlabel="s")
plot!(trained_solution[1,:], trained_solution[2,:],label="Structured O-NET", lw=2, trained_solution[3,:])




#################
# miscellaneous #
#################

# Prove that the model can substitute the resistive structure
input = (ode_data[2,:]./params[1]) .^2
dissipative_rate = Vector{Float64}()
for i in input
    dissipative_rate = push!(dissipative_rate, Structured_O_NET([i], θ, st)[1][2])
end
dissipative_energy = Vector{Float64}()
for i in input
    if dissipative_energy == Vector{Float64}()
        dissipative_energy  = push!(dissipative_energy, Structured_O_NET([i], θ, st)[1][2])
    else
        dissipative_energy  = push!(dissipative_energy, dissipative_energy[end]+Structured_O_NET([i], θ, st)[1][2])
    end
end

real_dissipative_rate = params[3]input
real_dissipative_energy = Vector{Float64}()
for i in real_dissipative_rate
    if real_dissipative_energy == Vector{Float64}()
        real_dissipative_energy  = push!(real_dissipative_energy, i)
    else
        real_dissipative_energy  = push!(real_dissipative_energy, real_dissipative_energy[end]+i)
    end
end

dissipative_rate
real_dissipative_rate
mechanical_energy = ode_data[1,:].^2/2params[2] + ode_data[2,:].^2/2params[1]
dissipative_energy
real_dissipative_energy
plot(time_steps, mechanical_energy, lw=2, label="Ground truth of mechanical energy", ylims=(0.0, 1.1), xlabel="Time step", ylabel="Energy/Exergy")
# dissipative_energy needs to be multiplied by 0.1, because the timesteps is 0.1
plot!(time_steps, real_dissipative_energy*0.1, lw=2, label="Ground truth of dissipative energy")
plot!(time_steps, dissipative_energy*0.1, lw=2, label="Prediction of dissipative energy")

plot(time_steps, real_dissipative_rate, lw=2, label="Ground truth of dissipative power", ylims=(0.0, 0.3), xlabel="Time step", ylabel="Power")
plot!(time_steps, dissipative_rate, lw=2, label="Prediction of dissipative power")





# Save and load parameters
using JLD2
JLD2.@save joinpath(@__DIR__, "results", "compositional_modelling_idho_params.jld") θ
JLD2.@load joinpath(@__DIR__, "results", "compositional_modelling_idho_params.jld") θ
θ