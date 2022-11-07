######################################
# Step 1: construct a neural network #
######################################

using Lux
H_NET = Lux.Chain(Lux.Dense(2, 20, tanh),
                  Lux.Dense(20, 10, tanh),
                  Lux.Dense(10, 1))

# "Random.default_rng" is a random number generator. It generates a random number in preparation for generating random parameters in the next code line.
using Random
rng = Random.default_rng()
# ps: the initial parameters of the neural network.
# st: the state of the neural network. It stores information (layers number, neurons number, activation function etc.) for reconstructing the neural network. For example, the output of the neural network with the given parameters is O_NET(x, ps, st)
ps, st = Lux.setup(rng, H_NET)




############################
# Step 2: construct an IVP #
############################

# FiniteDiff.jl is an automatic differentiation tool.
using FiniteDiff
function SymplecticGradient(NN, ps, st, z)
  # Compute the gradient of the neural network
  H = FiniteDiff.finite_difference_gradient(x -> sum(NN(x, ps, st)[1]), z)
  # Return the estimate of symplectic gradient
  return vec(cat(H[2:2, :], -H[1:1, :], dims=1))
end

function ODE(z, θ, t)
dz = SymplecticGradient(H_NET, θ, st, z)
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

# params = [m, c]
params = [2, 1] 
prob = ODEProblem(ODEFunction(ODEfunc_udho), initial_state, time_span, params)
ode_data = Array(CommonSolve.solve(prob, ImplicitMidpoint(), tstops = time_steps))

function solve_IVP(θ, batch_timesteps)
  IVP = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, (batch_timesteps[1], batch_timesteps[end]), θ)
  pred_data = Array(CommonSolve.solve(IVP, numerical_method, p=θ, tstops = batch_timesteps, sensealg=sensitivity_analysis))
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
using Flux: DataLoader
dataloader = DataLoader((ode_data, time_steps), batchsize = 200)

# Select an automatic differentiation tool
using Optimization
adtype = Optimization.AutoFiniteDiff()
# Construct an optimization problem with the given automatic differentiation and the initial parameters θ
optf = Optimization.OptimizationFunction((θ, ps, batch_data, batch_timesteps) -> loss_function(θ, batch_data, batch_timesteps), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(θ))
# Train the model multiple times. The "ncycle" is a function in the package IterTools.jl, it cycles through the dataloader "epochs" times.
using OptimizationOptimisers
using IterTools
epochs = 10;
result = Optimization.solve(optprob, Optimisers.ADAM(0.001), ncycle(dataloader, epochs), callback=callback)
# Access the trained parameters
θ = result.u

# The "loss_function" returns a tuple, where the first element of the tuple is the loss
loss = loss_function(θ, ode_data, time_steps)[1]

# Option: continue the training
include("helpers/train_helper.jl")
using Main.TrainInterface: LuxTrain
begin
  α = 0.0001
  epochs = 10
  θ = LuxTrain(optf, θ, α, epochs, dataloader, callback)
end


##########################
# Step 6: test the model #
##########################

H_NET(initial_state, θ, st)

# Plot phase portrait
trained_solution = CommonSolve.solve(IVP, numerical_method, p=θ, tstops = time_steps, sensealg=sensitivity_analysis)
using Plots
plot(ode_data[1,:], ode_data[2,:])
plot!(trained_solution[1,:], trained_solution[2,:])
