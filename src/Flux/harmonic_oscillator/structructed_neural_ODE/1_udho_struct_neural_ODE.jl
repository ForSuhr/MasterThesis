# undamped harmonic oscillator ODE

## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots
using BenchmarkTools
using ForwardDiff, Zygote

## initial condition
u0 = Float32[1.0, 1.0]
## set timespan
tspan = (1.0f0, 20.0f0)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize)
## set initial parameters
init_params = [1.0, 1.0]

## define an ODEFunction, which will generate the data we are trying to fit
function ODEfunc_udho(du,u,params,t) ### in-place usage
  ## conversion
  q, p = u
  m, c = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c
end

## define an ODEProblem with the self-defined ODEFunction above 
prob_ODEfunc_udho = ODEProblem(ODEfunc_udho, u0, tspan, init_params)
## solve the ODEProblem
sol = solve(prob_ODEfunc_udho, Tsit5(), saveat = tsteps)
## plot sol
plot(sol)
## pick out some data from sol
ode_data = Array(sol)
## plot original data (q,p)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Original")



## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
NN = FastChain(FastDense(2, 20, tanh), ### Multilayer perceptron for the part we don't know
           FastDense(20, 10, tanh),
           FastDense(10, 2))

## gradient of the neural network
dNN_dq = gradient(u -> NN(u, initial_params(NN))[1], u0)[1] ## ∂H/∂q
dNN_dp = gradient(u -> NN(u, initial_params(NN))[2], u0)[1] ## ∂H/∂p

## set neural ode problem
prob_neuralode = NeuralODE(dNN_dq, tspan, Tsit5(), saveat = tsteps)
## initial parameters for neural network
neural_params = prob_neuralode.p ## equivalent to initial_params(NN)

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
  end

## L2 loss function
function loss_neuralode(p)
      pred_data = predict_neuralode(p)
      loss = sum(abs2, ode_data .- pred_data) # Just sum of squared error, without mean
      return loss, pred_data
  end

## Callback function to observe training
callback = function(p, loss, pred_data)
    ### plot original and prediction data
    println(loss)
    x_axis_pred_data = pred_data[1,:]
    y_axis_pred_data = pred_data[2,:]
    plt = plot(x_axis_ode_data, y_axis_ode_data, label="Original")
    plot!(plt,x_axis_pred_data, y_axis_pred_data, label = "Prediction")
    display(plot(plt))
    return false
  end

## prob_neuralode.p are parameters, which are passed into loss_neuralode(p) as its argument
## optimizer chain of ADAM -> BFGS is used by default
@btime DiffEqFlux.sciml_train(loss_neuralode, dNN_dq, cb = callback, maxiters = 1)

