######################################
# Step 1: construct a neural network #
######################################

using Lux
Structured_ODE_NN = Lux.Chain(Lux.Dense(1, 20, tanh),
                             Lux.Dense(20, 10, tanh),
                             Lux.Dense(10, 1))

using Random
rng = Random.default_rng()
Random.seed!(rng, 0)
ps, st = Lux.setup(rng, Structured_ODE_NN)




############################
# Step 2: construct an IVP #
############################
m = 2
function ODE(dz, z, θ, t)
    q = z[1]
    p = z[2]
    dz[1] = p/m
    dz[2] = Structured_ODE_NN([q], θ, st)[1][1]
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
    println("loss: ", loss)
    return false
end




####################################
# Step 5: train the neural network #
####################################

# The dataloader generates a batch of data according to the given batchsize from the "training_data".
using Flux: DataLoader
begin
    using Flux: DataLoader
    time_steps_1 = range(0.0, 4.9, 50)
    time_steps_2 = range(0.0, 9.9, 100)
    time_steps_3 = range(0.0, 19.9, 200)
    dataloader1 = DataLoader((training_data[:,1:50], time_steps_1), batchsize = 50)
    dataloader2 = DataLoader((training_data[:,1:100], time_steps_2), batchsize = 100)
    dataloader3 = DataLoader((training_data[:,1:200], time_steps_3), batchsize = 200)
  end

begin
    include("helpers/train_helper.jl")
    using Main.TrainInterface: LuxTrain, OptFunction
    using Optimization
    optf = OptFunction(loss_function, Optimization.AutoZygote())
end
  
begin
    α = 0.002
    epochs = 100
    println("Training 1")
    θ = LuxTrain(optf, θ, α, epochs, dataloader1, callback)
end
  
begin
    α = 0.002
    epochs = 100
    println("Training 2")
    θ = LuxTrain(optf, θ, α, epochs, dataloader2, callback)
end
  
begin
    α = 0.001
    epochs = 100
    println("Training 3")
    θ = LuxTrain(optf, θ, α, epochs, dataloader3, callback)
end

# Save the parameters
begin
    using JLD2
    path = joinpath(@__DIR__, "parameters", "params_structured_ODE_NN.jld2")
    JLD2.save(path, "params_structured_ODE_NN", θ)
end

# Save the model
begin
    using JLD2
    path = joinpath(@__DIR__, "models", "structured_ODE_NN.jld2")
    JLD2.save(path, "structured_ODE_NN", Structured_ODE_NN, "st", st)
end




##########################
# Step 6: test the model #
##########################

# Load the parameters
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "parameters", "params_structured_ODE_NN.jld2")
    θ = JLD2.load(path, "params_structured_ODE_NN")
end

# Load the model
begin
    using JLD2, Lux
    path = joinpath(@__DIR__, "models", "structured_ODE_NN.jld2")
    Structured_ODE_NN = JLD2.load(path, "structured_ODE_NN")
    st = JLD2.load(path, "st")
end

Structured_ODE_NN([initial_state[1]], θ, st)[1][1]

# Plot phase portrait
begin
    IVP_test = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, time_span_total, θ)
    predict_data = CommonSolve.solve(IVP_test, numerical_method, p=θ, tstops = time_steps_total, sensealg=sensitivity_analysis)
    using Plots
    plot(ode_data[1,:], ode_data[2,:], lw=3, xlabel="q", ylabel="p")
    plot!(predict_data[1,:], predict_data[2,:], lw=3)
end
