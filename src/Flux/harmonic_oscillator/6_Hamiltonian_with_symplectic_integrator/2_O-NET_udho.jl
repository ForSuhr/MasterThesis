## using package
using DiffEqFlux, DifferentialEquations, Plots
using Optimization, OptimizationFlux
using BenchmarkTools
using Noise
using IterTools: ncycle
using ReverseDiff

## define ODEs
function ODEfunc_udho(du,u,params,t)
  ## conversion
  q, p = u
  m, c = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c
end

## give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0; 1.0]
tspan = (0.0, 19.9)
tsteps = range(tspan[1], tspan[2], length = 200)
init_params = [2.0, 1.0]
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
#sol = solve(prob, ImplicitMidpoint(), tstops = tsteps)
sol = solve(prob, Midpoint(), saveat = tsteps)
## print origin data
ode_data = Array(sol)
# ode_data = add_gauss(ode_data, 0.01)
q_ode_data = ode_data[1,:]
p_ode_data = ode_data[2,:]
plt = plot(q_ode_data, p_ode_data, lw=3, label="Ground truth")


NN = Flux.Chain(Flux.Dense(2, 40, tanh),
           Flux.Dense(40, 40, tanh),
           Flux.Dense(40, 2))
### check the parameters prob_neuralode.p in prob_neuralode
neural_params = prob_neuralode.p

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p, timesteps)
  prob_neuralode = DiffEqFlux.NeuralODE(NN, tspan, Midpoint(), saveat = timesteps)
  Array(prob_neuralode(u0, p))
end

## L2 loss function
function loss_neuralode(p, batch_data, timesteps)
    pred_data = predict_neuralode(p, timesteps)
    loss = sum(abs2, batch_data .- pred_data)
    return loss ,pred_data
end


## Callback function to observe training
callback = function(params, loss, pred_data)
    prob_neuralode = DiffEqFlux.NeuralODE(NN, tspan, Midpoint(), saveat = tsteps)
    pred_data = Array(prob_neuralode(u0, params))
    total_loss = sum(abs2, ode_data .- pred_data)
    println(total_loss)
    if loss > 0.001 
        return false
      else
        return true
      end
end

# Use mini-batch gradient descent
dataloader = Flux.Data.DataLoader((ode_data, tsteps), batchsize = 100)
loss_neuralode(neural_params, dataloader.data[1], dataloader.data[2])
epochs = 10

## first round of training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((θ, ps, batch_data, timesteps) -> loss_neuralode(θ, batch_data, timesteps), adtype)
optprob1 = Optimization.OptimizationProblem(optf, neural_params)
@time res1 = Optimization.solve(optprob1, Optimisers.ADAM(0.01), ncycle(dataloader, epochs), callback = callback)
## second round of training
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
@time res2 = Optimization.solve(optprob2, Optimisers.ADAM(0.001), ncycle(dataloader, epochs), callback = callback)
## third round of training
optprob3 = Optimization.OptimizationProblem(optf, res2.u)
@time res3 = Optimization.solve(optprob3, Optimisers.ADAM(0.0001), ncycle(dataloader, epochs), callback = callback)
params_O_NET = res3.u

# repeat training
optprob4 = Optimization.OptimizationProblem(optf, params_O_NET)
@time res4 = Optimization.solve(optprob4, ADAM(0.0001), callback = callback, maxiters = 500)
params_O_NET = res4.u

## check phase portrait
prob_neuralode = DiffEqFlux.NeuralODE(NN, tspan, Midpoint(), saveat = timesteps)
trajectory_estimate_O_NET = Array(prob_neuralode(u0, params_O_NET))
plt = plot(q_ode_data, p_ode_data, xlims=(-2,3), ylims=(-2,3), lw=5, label="Ground truth", linestyle=:solid, xlabel="q", ylabel="p")
plt = plot!(trajectory_estimate_O_NET[1,:], trajectory_estimate_O_NET[2,:], lw=5, label = "O-NET", linestyle=:dashdot)

# Calculate training loss
training_error_O_NET = sum((ode_data .- trajectory_estimate_O_NET).^2)/2/200
begin
min_error = (findmax(sum((ode_data[:, 1:20] .- trajectory_estimate_O_NET[:, 1:20]).^2, dims=1)/2)[1] +
            findmin(sum((ode_data[:, 1:20] .- trajectory_estimate_O_NET[:, 1:20]).^2, dims=1)/2)[1])/2
max_error = (findmax(sum((ode_data[:, :] .- trajectory_estimate_O_NET[:, :]).^2, dims=1)/2)[1] +
            findmin(sum((ode_data[:, :] .- trajectory_estimate_O_NET[:, :]).^2, dims=1)/2)[1])/2
end
(min_error + max_error)/2
max_error - (min_error + max_error) /2
# Calculate test loss
test_tspan = (20.0, 39.9)
test_tsteps = range(test_tspan[1], test_tspan[2], length = 200)
test_prob = ODEProblem(ODEfunc_udho, u0, test_tspan, init_params)
test_sol = solve(test_prob, ImplicitMidpoint(), tstops = test_tsteps)
test_data = Array(test_sol)
test_data = add_gauss(test_data, 0.01)
q_test_data = test_data[1,:]
p_test_data = test_data[2,:]
test_prob_neuralode = DiffEqFlux.NeuralODE(NN, test_tspan, Midpoint(), saveat = test_tsteps)
test_trajectory_estimate_O_NET = Array(test_prob_neuralode(u0, params_O_NET))
test_error_O_NET = sum((test_data .- test_trajectory_estimate_O_NET).^2)/2/200
begin
min_error = (findmax(sum((test_data[:, 1:20] .- test_trajectory_estimate_O_NET[:, 1:20]).^2, dims=1)/2)[1] +
            findmin(sum((test_data[:, 1:20] .- test_trajectory_estimate_O_NET[:, 1:20]).^2, dims=1)/2)[1])/2
max_error = (findmax(sum((test_data[:, :] .- test_trajectory_estimate_O_NET[:, :]).^2, dims=1)/2)[1] +
            findmin(sum((test_data[:, :] .- test_trajectory_estimate_O_NET[:, :]).^2, dims=1)/2)[1])/2
end
(min_error + max_error)/2
max_error - (min_error + max_error) /2

# check L2 test error
l2_error_O_NET = vec(sum((test_data .- test_trajectory_estimate_O_NET).^2, dims=1)/2)
plt = plot(tsteps, l2_error_O_NET, lw=3, label="O-NET", ylims=(0,0.0013), xlabel="Time step", ylabel="Prediction error of coordinates")

# check Hamiltonian
H_ground_truth = ode_data[2,:].^2/(2*init_params[1]) + ode_data[1,:].^2/(2*init_params[2])
H_O_NET = trajectory_estimate_O_NET[2,:].^2/(2*init_params[1]) + trajectory_estimate_O_NET[1,:].^2/(2*init_params[2])
plt = plot(tsteps, H_ground_truth, lw=3, label="Ground truth")
plt = plot!(tsteps, H_O_NET, lw=3, label="O-NET", ylabel="Hamiltonian", xlabel="Time step")
(- findmin(H_O_NET - H_ground_truth)[1] + findmax(H_O_NET - H_ground_truth)[1])/2
test_H_ground_truth = test_data[2,:].^2/(2*init_params[1]) + test_data[1,:].^2/(2*init_params[2])
test_H_O_NET = test_trajectory_estimate_O_NET[2,:].^2/(2*init_params[1]) + test_trajectory_estimate_O_NET[1,:].^2/(2*init_params[2])
plt = plot(test_tsteps, test_H_ground_truth, lw=3, label="Ground truth")
plt = plot!(test_tsteps, test_H_O_NET, lw=3, label="O-NET", ylabel="Hamiltonian", xlabel="Time step")
(- findmin(test_H_O_NET - test_H_ground_truth)[1] + findmax(test_H_O_NET - test_H_ground_truth)[1])/2


## save the parameters
using JLD
save(joinpath(@__DIR__, "results", "params_O_NET.jld"), "params_O_NET", params_O_NET)
params = load(joinpath(@__DIR__, "results", "params_O_NET.jld"), "params_O_NET")
params_O_NET = params
