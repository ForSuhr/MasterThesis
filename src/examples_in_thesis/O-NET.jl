######################################
# Step 1: construct a neural network #
######################################

using Flux
# Dense: construct a layer. For instance, Dense(2, 40, tanh) constructs a 
# 2-input and 40-output layer with the activation function tanh.
# Chain: connect layers.
# O_NET: a feedforward neural network with 2 neurons in the input layer, 
# 40 neurons in the first hidden layer, 40 neurons in the second hidden layer 
# and 2 neurons in the output layer.
O_NET = Flux.Chain(Flux.Dense(2, 20, tanh),
                    Flux.Dense(20, 10, tanh),
                    Flux.Dense(10, 2))
# ps: the initial parameters of the neural network. 
# re: a method to reconstruct the neural network with the given 
# parameters ps and input x, e.g., re(ps)(x) is the output of the 
# neural network with the given parameters ps and input x.
ps, re = Flux.destructure(O_NET)




############################
# Step 2: construct an IVP #
############################

# dz is the time derivative of z at a fixed time.
# Note: the "t" in the argument is designed for nonautonomous case. 
# This "t" is not the same concept as timesteps. 
# In the case of Hamiltonian (autonomous), this "t" will not be used.
function ODE(dz, z, θ, t)
    # In Flux.jl, re(θ)(z) is the output of O-NET with the given parameters θ and input z. 
    # In Lux.jl, this term should be rewritten as O_NET(z, θ, st).
    dz[1] = re(θ)(z)[1]
    dz[2] = re(θ)(z)[2]
end

# initial state
initial_state = [1.0, 1.0]
    
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

function ODEfunc_udho(dz, z, params, t)
    q, p = z
    m, c = params
    dz[1] = p/m
    dz[2] = -q/c
end

# params = [m, c]
params = [2, 1] 
prob = ODEProblem(ODEFunction(ODEfunc_udho), initial_state, time_span, params)
ode_data = Array(CommonSolve.solve(prob, ImplicitMidpoint(), tstops = time_steps))

function solve_IVP(θ, batch_timesteps)
    IVP = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, (batch_timesteps[1], batch_timesteps[end]), θ)
    pred_data = Array(CommonSolve.solve(IVP, Midpoint(), p=θ, saveat = batch_timesteps, sensealg=sensitivity_analysis))
    return pred_data
end

function loss_function(θ, batch_data, batch_timesteps)
    pred_data = solve_IVP(θ, batch_timesteps)
    # "batch_data" is a batch of ode data
    loss = sum((batch_data .- pred_data) .^ 2)
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

dataloader = Flux.Data.DataLoader((ode_data, time_steps), batchsize = 200)

# Select an automatic differentiation tool
using Optimization
adtype = Optimization.AutoZygote()
# Construct an optimization problem with the given automatic differentiation and the initial parameters θ
optf = Optimization.OptimizationFunction((θ, ps, batch_data, batch_timesteps) -> loss_function(θ, batch_data, batch_timesteps), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)
# Train the model multiple times. The "ncycle" is a function in the package IterTools.jl, it cycles through the dataloader "epochs" times.
using OptimizationOptimisers
using IterTools
epochs = 10;
result = Optimization.solve(optprob, Optimisers.ADAM(0.001), ncycle(dataloader, epochs), callback=callback)
# Access the trained parameters
θ = result.u

# The "loss_function" returns a tuple, where the first element of the tuple is the loss
loss = loss_function(result.u, ode_data, time_steps)[1]

# Option: continue the training
include("helpers/train_helper.jl")
using Main.TrainInterface: FluxTrain
θ = FluxTrain(optf, θ, 0.0001, 100, dataloader, callback)




##########################
# Step 6: test the model #
##########################

# Recall that "re" is a method to reconstruct the neural network.
re(θ)(initial_state)

# Plot phase portrait
trained_solution = CommonSolve.solve(IVP, numerical_method, p=θ, tstops = time_steps, sensealg=sensitivity_analysis)
using Plots
plot(ode_data[1,:], ode_data[2,:], xlabel="q", ylabel="p")
plot!(trained_solution[1,:], trained_solution[2,:])

# Save the parameters
using JLD
path = joinpath(@__DIR__, "results", "params_O_NET.jld")
JLD.save(path, "params_O_NET", θ)
θ = JLD.load(path, "params_O_NET")
