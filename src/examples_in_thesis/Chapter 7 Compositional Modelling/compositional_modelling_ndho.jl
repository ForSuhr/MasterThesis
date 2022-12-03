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
begin
m = 1
c = 1
θ_0 = 20
θ_d = 30
α = 1.5
function ODE(dz, z, θ, t)
    q, p, sd, sₑ = z
    v = p/m
    dz[1] = v
    dz[2] = -q/c + Structured_O_NET([v], θ, st)[1][1]
    dz[3] = Structured_O_NET([v^2], θ, st)[1][2] / θ_0 - α*(θ_d-θ_0)/θ_d
    dz[4] = α*(θ_d-θ_0)/θ_0
end
end

# initial state
initial_state = [1.0, 1.0, 5.0, 10.0]
    
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

function ODEfunc_ndho(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
    q, p, s_d, s_e = u
    m, c, d, θ_0, θ_d, α = params
    v = p/m
    ## ODEs
    du[1] = v
    du[2] = -q/c-d*v
    du[3] = d*((v)^2)/θ_d-α*(θ_d-θ_0)/θ_d
    du[4] = α*(θ_d-θ_0)/θ_0
end

# params = [m, c]
params = [1.0, 1.0, 0.5, 20, 30, 1.5] 
prob = ODEProblem(ODEFunction(ODEfunc_ndho), initial_state, time_span, params)
ode_data = Array(CommonSolve.solve(prob, ImplicitMidpoint(), tstops = time_steps))
using Plots
plot(ode_data[1,:], ode_data[2,:], xlabel="q", ylabel="p")
plot!(pred_data[1,:], pred_data[2,:], xlabel="q", ylabel="p")


function solve_IVP(θ, batch_timesteps)
    IVP = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, (batch_timesteps[1], batch_timesteps[end]), θ)
    #pred_data = Array(CommonSolve.solve(IVP, Midpoint(), p=θ, saveat = batch_timesteps, sensealg=sensitivity_analysis))
    pred_data = Array(CommonSolve.solve(IVP, ImplicitMidpoint(), p=θ, tstops = batch_timesteps, sensealg=sensitivity_analysis))
    return pred_data
end

function loss_function(θ, batch_data, batch_timesteps)
    pred_data = solve_IVP(θ, batch_timesteps)
    #loss = sum((batch_data[:,1:length(pred_data[1,:])] .- pred_data) .^ 2)
    loss = sum((batch_data .- pred_data) .^ 2)
    return loss, pred_data
end

callback = function(θ, loss, pred_data)
    plt = plot(ode_data[1,:], ode_data[2,:], label="Ground truth")
    plot!(plt, pred_data[1,:], pred_data[2,:], label = "Prediction")
    display(plot(plt))
    println("loss: ", loss_function(θ, ode_data, time_steps)[1])
    return false
end

solve_IVP(θ, time_steps)
loss_function(θ, ode_data, time_steps)[1]




####################################
# Step 5: train the neural network #
####################################

# The dataloader generates a batch of data according to the given batchsize from the "ode_data".
using Flux: DataLoader
#dataloader = DataLoader((ode_data, dz_data), batchsize = 200)
dataloader = DataLoader((ode_data, time_steps), batchsize = 200)

# Select an automatic differentiation tool
using Optimization
#adtype = Optimization.AutoFiniteDiff()
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
loss = loss_function(θ, ode_data, time_steps)[1]

# Option: continue the training
include("helpers/train_helper.jl")
using Main.TrainInterface: LuxTrain
# Adjust the learning rate and epochs, then repeat this code block
begin
    learning_rate = 0.001
    epochs = 10
    θ = LuxTrain(optf, θ, learning_rate, epochs, dataloader, callback)
end




##########################
# Step 6: test the model #
##########################

Structured_O_NET([initial_state[2]/m], θ, st)[1][1]

# Plot phase portrait
begin
IVP = SciMLBase.ODEProblem(ODEFunction(ODE), initial_state, time_span, θ)
trained_solution = CommonSolve.solve(IVP, ImplicitMidpoint(), p=θ, tstops = time_steps, sensealg=sensitivity_analysis)
plot(ode_data[1,:], ode_data[2,:], xlabel="q", ylabel="p", xlims=(-2,2), ylims=(-2,2))
plot!(trained_solution[1,:], trained_solution[2,:])
end


#################
# miscellaneous #
#################

# Save and load parameters
using JLD2
JLD2.@save joinpath(@__DIR__, "results", "params.jld") θ
JLD2.@load joinpath(@__DIR__, "results", "params.jld") θ
θ