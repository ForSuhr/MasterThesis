# undamped harmonic oscillator ODE
## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots
using BenchmarkTools
## using CUDA

## initial condition
x0 = Float32[1.0;0.0]
## set timespan
datasize = 1000
tspan = (1.0f0, 20.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)


## define an ODEFunction, which will generate the data we are trying to fit
function ODEfunc_uho(df,x,p,t) ### in-place usage
  q = x[1] ### q is the displacement of the spring
  p = x[2] ### p is the momentum of the mass
  c = 0.5  ### c is the spring compliance   
  m = 1 ### m is the mass
  v = p/m ### v is the velocity of the mass
  dq = v
  dp = -q/c
  df[1] = dq
  df[2] = dp
end

## define an ODEProblem with the self-defined ODEFunction above 
prob_ODEfunc_uho = ODEProblem(ODEfunc_uho, x0, tspan, 0)

## solve the ODEProblem
sol = solve(prob_ODEfunc_uho, Tsit5(), saveat = tsteps)

## plot sol
plot(sol)
## pick out some data from sol
ode_data = Array(sol)
## plot original data (q,p)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Original")



## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
dudt2 = FastChain(FastDense(2, 20, tanh), ### Multilayer perceptron for the part we don't know
                  FastDense(20, 10, tanh),
                  FastDense(10, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
### check the parameters prob_neuralode.p in prob_neuralode
prob_neuralode.p

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
    Array(prob_neuralode(x0, p))
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
@btime DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, cb = callback)    



