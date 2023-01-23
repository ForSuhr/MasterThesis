######################################
# Step 1: construct a neural network #
######################################

using Lux
HNN = Lux.Chain(Lux.Dense(2, 20, tanh),
                  Lux.Dense(20, 10, tanh),
                  Lux.Dense(10, 1))

# "Random.default_rng" is a random number generator. It generates a random number in preparation for generating random parameters in the next code line.
using Random
rng = Random.default_rng()
# ps: the initial parameters of the neural network.
# st: the state of the neural network. It stores information (layers number, neurons number, activation function etc.) for reconstructing the neural network. For example, the output of the neural network with the given parameters is O_NET(x, ps, st)
ps, st = Lux.setup(rng, HNN)




############################
# Step 2: construct an IVP #
############################

# FiniteDiff.jl is an automatic differentiation tool.
using FiniteDiff
function SymplecticGradient(NN, ps, st, z)
  # Compute the gradient of the neural network
  ∂H = FiniteDiff.finite_difference_gradient(x -> sum(NN(x, ps, st)[1]), z)
  # Return the estimate of symplectic gradient
  return cat(∂H[2:2, :], -∂H[1:1, :], dims=1)
end

function ODE(z, θ, t)
dz = vec(SymplecticGradient(HNN, θ, st, z))
end

# initial state
initial_state = [1.0, 1.0]
    
# Starting at 0.0 and ending at 19.9, the length of each step is 0.1. Thus, we have 200 time steps in total.
time_span = (0.0, 19.9)
time_steps = range(0.0, 19.9, 200)

# parameters of the neural network
θ = ps

# ODEProblem is a IVP constructor in the Julia package SciMLBase.jl
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
sensitivity_analysis = InterpolatingAdjoint(autojacvec=ZygoteVJP(true))

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

# mass m and spring compliance c
params = [2, 1]
# Generate data set
time_span_total = (0.0, 24.9)
time_step_number_total = 250
time_steps_total = range(0.0, 24.9, time_step_number_total)
prob = ODEProblem(ODEFunction(ODEfunc_udho), initial_state, time_span_total, params)
ode_data = Array(CommonSolve.solve(prob, ImplicitMidpoint(), tstops = time_steps_total))
# Split data set into training and test sets, 80% and 20% respectively
training_data = ode_data[:, 1:Int(time_step_number_total*0.8)]
test_data = ode_data[:, 1:Int(time_step_number_total*0.2)]

# Generate the time derivatives of the coordinates
dq_data = training_data[2,:]/params[1]
dp_data = -training_data[1,:]/params[2]
dq_data = reshape(dq_data, 1, :)
dp_data = reshape(dp_data, 1, :)
dz_data = cat(dq_data, dp_data, dims = 1)

function loss_function(θ, batch_data, batch_dz_data)
  pred_data = SymplecticGradient(HNN, θ, st, batch_data)
  loss = sum((batch_dz_data .- pred_data) .^ 2)
  return loss, pred_data
end

callback = function(θ, loss, pred_data)
  println(loss_function(θ, training_data, dz_data)[1])
  return false
end




####################################
# Step 5: train the neural network #
####################################

# The dataloader generates a batch of data according to the given batchsize from the "training_data".
using Flux: DataLoader
# (training_data, dz_data) is the whole training set.
dataloader = DataLoader((training_data, dz_data), batchsize = 50)

# Select an automatic differentiation tool
using Optimization
adtype = Optimization.AutoFiniteDiff()
# Construct an optimization problem with the given automatic differentiation and the initial parameters θ
optf = Optimization.OptimizationFunction((θ, ps, batch_data, batch_dz_data) -> loss_function(θ, batch_data, batch_dz_data), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(θ))
# Train the model multiple times. The "ncycle" is a function in the package IterTools.jl, it cycles through the dataloader "epochs" times.
begin
  using OptimizationOptimisers
  using IterTools
  epochs = 10;
  result = Optimization.solve(optprob, Optimisers.ADAM(0.01), ncycle(dataloader, epochs), callback=callback)
  # Access the trained parameters
  θ = result.u
end

# The "loss_function" returns a tuple, where the first element of the tuple is the loss
loss = loss_function(θ, training_data, dz_data)[1]

# Option: continue the training
include("helpers/train_helper.jl")
using Main.TrainInterface: LuxTrain
begin
  α = 0.001
  epochs = 10
  θ = LuxTrain(optf, θ, α, epochs, dataloader, callback)
end

# Save the parameters
begin
  using JLD2
  path = joinpath(@__DIR__, "parameters", "params_HNN.jld2")
  JLD2.save(path, "params_HNN", θ)
end

# Save the model
begin
  using JLD2
  path = joinpath(@__DIR__, "models", "HNN.jld2")
  JLD2.save(path, "HNN", HNN, "st", st)
end




##########################
# Step 6: test the model #
##########################

# Load the parameters
begin
  using JLD2, Lux
  path = joinpath(@__DIR__, "parameters", "params_HNN.jld2")
  θ = JLD2.load(path, "params_HNN")
end

# Load the model
begin
  using JLD2, Lux
  path = joinpath(@__DIR__, "models", "HNN.jld2")
  HNN = JLD2.load(path, "HNN")
  st = JLD2.load(path, "st")
end

HNN(initial_state, θ, st)

# Plot phase portrait
IVP_test = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, time_span_total, θ)
predict_data = CommonSolve.solve(IVP_test, numerical_method, p=θ, tstops = time_steps_total, sensealg=sensitivity_analysis)
using Plots
plot(ode_data[1,:], ode_data[2,:], lw=3, xlabel="q", ylabel="p")
plot!(predict_data[1,:], predict_data[2,:], lw=3)
