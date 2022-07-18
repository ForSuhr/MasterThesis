# https://docs.juliahub.com/DiffEqFlux/BdO4p/1.20.0/examples/feedback_control/
# https://docs.juliahub.com/DiffEqFlux/BdO4p/1.20.0/examples/optimal_control/

## use packages
using DiffEqFlux, DifferentialEquations, LinearAlgebra
using Optimization, OptimizationFlux
using Plots

## define ODEs
function ODEfunc_ndho(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
  q, p, sd, sₑ = u
  m, d, c, θₒ, θd, α = init_params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c-d*p/m
  du[3] = d*((p/m)^2)/θd-α*(θd-θₒ)/θd
  du[4] = α*(θd-θₒ)/θₒ
end

## give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0, 1.0, 0.0, 0.0]
tspan = (0.0, 20.0)
datasize = 100
tsteps = collect(range(tspan[1], tspan[2], length = datasize))
init_params = [1.0; 0.4; 1.0; 2.0; 3.0; 5.0]
prob = ODEProblem(ODEfunc_ndho, u0, tspan, init_params)

## solve the ODE problem
sol = solve(prob, Tsit5(), saveat = tsteps)

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
z1_axis_ode_data = ode_data[3,:]
z2_axis_ode_data = ode_data[4,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, z1_axis_ode_data, label="Ground truth")
plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_ode_data, label="Ground truth")

## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
NN = FastChain(FastDense(2, 20, tanh), ### Multilayer perceptron for the part we don't know
                  FastDense(20, 10, tanh),
                  FastDense(10, 2))
# prob_neuralode = NeuralODE(NN, tspan, Tsit5(), saveat = tsteps)
### check the parameters prob_neuralode.p in prob_neuralode
neural_params = initial_params(NN)


NN
# The model weights are destructured into a vector of parameters
size_neural_params = length(neural_params)
zeros_params = zeros(size_neural_params)
## the first output of the NN
NN(u0[3:4], zeros_params)

## replace a part of the ODEs with a neural network
function ODEfunc_ndho_pred(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
  q, p, sd, sₑ = u
  m, d, c, θₒ, θd, α = init_params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c-d*p/m
  # du[3] = d*((p/m)^2)/θd-α*(θd-θₒ)/θd
  # du[4] = α*(θd-θₒ)/θₒ
  du[3] = d*((p/m)^2)/θd + NN(0, params[1:size_neural_params])[1]
  du[4] = NN(0, params[1:size_neural_params])[2]
end

## construct an ODE problem with NN replacement part
prob_pred = ODEProblem(ODEfunc_ndho_pred, u0, tspan, init_params)

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(params)
  Array(solve(prob_pred, Tsit5(), p=params, saveat=tsteps,
        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

## L2 loss function
function loss_neuralode(params)
    pred_data = predict_neuralode(params)
    # rss = sum(abs2, ode_data .- pred_data)
    error = sum(abs2, ode_data[[3,4],:] .- pred_data[[3,4],:]) # Just sum of squared error, without mean
    return error, pred_data
end

## Callback function to observe training
callback = function(params, loss, pred_data)
  ### plot Ground truth and prediction data
  println(loss)
  x_axis_pred_data = pred_data[1,:]
  y_axis_pred_data = pred_data[2,:]
  z1_axis_pred_data = pred_data[3,:]
  z2_axis_pred_data = pred_data[4,:]
  plt = plot(x_axis_ode_data, y_axis_ode_data, z1_axis_ode_data, label="Ground truth")
  plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_ode_data, label="Ground truth")
  plt = plot!(x_axis_ode_data, y_axis_ode_data, z1_axis_pred_data, label = "Prediction")
  plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_pred_data, label = "Prediction")
  display(plot(plt))

  if loss > 0.1 
    return false
  else
    return true
  end
end



## training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, neural_params)
res1 = Optimization.solve(optprob, ADAM(0.05), callback = callback, maxiters = 300)


optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, ADAM(0.01), callback = callback,maxiters = 300)

optprob3 = Optimization.OptimizationProblem(optf, res2.u)
res3 = Optimization.solve(optprob3, ADAM(0.001), callback = callback,maxiters = 300)


ode_data
data_pred = predict_neuralode(res2.u)
plt = plot(ode_data[1,:], ode_data[2,:], label = "Ground truth")
plt = plot!(data_pred[1,:], data_pred[2,:], label = "Prediction")