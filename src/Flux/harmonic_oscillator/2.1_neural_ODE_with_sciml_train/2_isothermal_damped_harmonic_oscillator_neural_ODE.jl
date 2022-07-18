# isothermal damped harmonic oscillator ODE
## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots
using BenchmarkTools
## using CUDA

## initial condition
x0 = Float32[1.0;0.0;0.0]
## set timespan
datasize = 1000
tspan = (1.0f0, 20.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)


## define ODEs
function ODEfunc_idho(du,u,params,t) ### du=[̇q,̇p,̇sₑ], u=[q,p,sₑ], params=[m,d,θₒ,c]
  q, p, sₑ = u
  m, d, θₒ, c = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c-d*p/m
  du[3] = d*(p/m)^2/θₒ
end

"""
m is the mass
c is the spring compliance
d is the damping coefficient
θ_o is the environmental temperature
q is the displacement of the spring
p is the momentum of the mass
s_e is the entropy of the environment
"""

## give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0, 1.0, 0.0]
tspan = (0.0, 20.0)
tsteps = range(tspan[1], tspan[2], length = 1000)
init_params = [1.0, 0.4, 1.0, 1.0] ### parameters = [m,d,θₒ,c]
prob = ODEProblem(ODEfunc_idho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, Tsit5(), saveat = tsteps)

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
z_axis_ode_data = ode_data[3,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth")



## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
NN = FastChain(FastDense(3, 20, tanh), ### Multilayer perceptron for the part we don't know
                  FastDense(20, 10, tanh),
                  FastDense(10, 3))
prob_neuralode = NeuralODE(NN, tspan, Tsit5(), saveat = tsteps)
### check the parameters prob_neuralode.p in prob_neuralode
neural_params = prob_neuralode.p

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
  end

## L2 loss function
function loss_neuralode(p)
      pred_data = predict_neuralode(p)
      loss = mean(sum(abs2, ode_data .- pred_data)) ### mean squared error
      return loss, pred_data
  end

## Callback function to observe training
callback = function(p, loss, pred_data)
    ### plot Ground truth and prediction data
    println(loss)
    x_axis_pred_data = pred_data[1,:]
    y_axis_pred_data = pred_data[2,:]
    z_axis_pred_data = pred_data[3,:]
    plt = plot(x_axis_ode_data, y_axis_ode_data, z_axis_ode_data, label="Ground truth")
    plot!(plt,x_axis_pred_data, y_axis_pred_data, z_axis_pred_data, label = "Prediction")
    display(plot(plt))
    return false
  end

## prob_neuralode.p are parameters, which are passed into loss_neuralode(p) as its argument
@btime DiffEqFlux.sciml_train(loss_neuralode, neural_params, cb = callback)    



