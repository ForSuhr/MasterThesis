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
Random.seed!(rng, 0)
# ps: the initial parameters of the neural network.
# st: the state of the neural network. It stores information (layers number, neurons number, activation function etc.) for reconstructing the neural network. For example, the output of the neural network with the given parameters is O_NET(x, ps, st)
ps, st = Lux.setup(rng, H_NET)




############################
# Step 2: construct an IVP #
############################

# FiniteDiff.jl is an automatic differentiation tool.
using FiniteDiff
using ReverseDiff
function SymplecticGradient(NN, ps, st, z)
  # Compute the gradient of the neural network
  ∂H = FiniteDiff.finite_difference_gradient(x -> sum(NN(x, ps, st)[1]), z)
  # ∂H = ReverseDiff.gradient(x -> sum(NN(x, ps, st)[1]), z)
  # Return the estimate of symplectic gradient
  return vec(cat(∂H[2:2, :], -∂H[1:1, :], dims=1))
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
numerical_method = Tsit5()

# Select the adjoint method to computer the gradient of the loss with respect to the parameters. 
# ReverseDiffVJP is a callable function in the package SciMLSensitivity.jl, it uses the automatic 
# differentiation tool ReverseDiff.jl to compute the vector-Jacobian products (VJP) efficiently. 
using ReverseDiff
using SciMLSensitivity
sensitivity_analysis = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

# Use the ODE Solver CommonSolve.solve to yield solution. And the solution is the estimate of the coordinates trajectories.
using CommonSolve
solution = CommonSolve.solve(IVP, numerical_method, p=θ, saveat = time_steps, sensealg=sensitivity_analysis)

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


function solve_IVP(θ, batch_timesteps)
  IVP = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, (batch_timesteps[1], batch_timesteps[end]), θ)
  pred_data = Array(CommonSolve.solve(IVP, numerical_method, p=θ, saveat = batch_timesteps, sensealg=sensitivity_analysis))
  return pred_data
end

function loss_function(θ, batch_data, batch_timesteps)
  pred_data = solve_IVP(θ, batch_timesteps)
  # "batch_data" is a batch of ode data
  loss = sum((batch_data .- pred_data) .^ 2)
  return loss, pred_data
end

callback = function(θ, loss, pred_data)
  println("loss: ", loss)
  return false
end


####################################
# Step 5: train the neural network #
####################################
# The dataloader generates a batch of data according to the given batchsize from the "ode_data".
begin
  using Flux: DataLoader
  time_steps_1 = range(0.0, 4.9, 50)
  time_steps_2 = range(0.0, 9.9, 100)
  time_steps_3 = range(0.0, 19.9, 200)
  dataloader1 = DataLoader((training_data[:,1:50], time_steps_1), batchsize = 50)
  dataloader2 = DataLoader((training_data[:,1:100], time_steps_2), batchsize = 100)
  dataloader3 = DataLoader((training_data[:,1:200], time_steps_3), batchsize = 200)
end

# The "loss_function" returns a tuple, where the first element of the tuple is the loss
loss = loss_function(θ, training_data, time_steps)[1]

begin
  include("helpers/train_helper.jl")
  using Main.TrainInterface: LuxTrain, OptFunction
  using Optimization
  optf = OptFunction(loss_function, Optimization.AutoFiniteDiff())
  # optf = OptFunction(loss_function, Optimization.AutoReverseDiff())
end

# Adjust the learning rate and repeat training by using a increasing time span strategy to escape from local minima
# Please refer to https://docs.juliahub.com/DiffEqSensitivity/02xYn/6.78.2/training_tips/local_minima/
begin
  α = 0.002
  epochs = 300
  println("Training 1")
  θ = LuxTrain(optf, θ, α, epochs, dataloader1, callback)
end

begin
  α = 0.002
  epochs = 300
  println("Training 2")
  θ = LuxTrain(optf, θ, α, epochs, dataloader2, callback)
end

begin
  α = 0.002
  epochs = 100
  println("Training 3")
  θ = LuxTrain(optf, θ, α, epochs, dataloader3, callback)
end

# Save the parameters
begin
  using JLD2
  path = joinpath(@__DIR__, "parameters", "params_H_NET.jld2")
  JLD2.save(path, "params_H_NET", θ)
end

# Save the model
begin
  using JLD2
  path = joinpath(@__DIR__, "models", "H_NET.jld2")
  JLD2.save(path, "H_NET", H_NET, "st", st)
end




##########################
# Step 6: test the model #
##########################

# Load the parameters
begin
  using JLD2, Lux
  path = joinpath(@__DIR__, "parameters", "params_H_NET.jld2")
  θ = JLD2.load(path, "params_H_NET")
end

# Load the model
begin
  using JLD2, Lux
  path = joinpath(@__DIR__, "models", "H_NET.jld2")
  H_NET = JLD2.load(path, "H_NET")
  st = JLD2.load(path, "st")
end

H_NET(initial_state, θ, st)

# Plot phase portrait
IVP_test = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, time_span_total, θ)
predict_data = CommonSolve.solve(IVP_test, numerical_method, p=θ, tstops = time_steps_total, sensealg=sensitivity_analysis)
using Plots
plot(ode_data[1,:], ode_data[2,:], lw=3, xlabel="q", ylabel="p", label="Ground truth")
plot!(predict_data[1,:], predict_data[2,:], lw=3, label="H-NET")
