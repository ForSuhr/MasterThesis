# nonisothermal damped harmonic oscillator ODE
## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots
using BenchmarkTools

## define ODEs
function ODEfunc_ndho(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
  q, p, sd, sₑ = u
  m, d, c, θₒ, θd, α = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c-d*p/m
  du[3] = d*((p/m)^2)/θd-α*(θd-θₒ)/θd
  du[4] = α*(θd-θₒ)/θₒ
end

"""
m is the mass
c is the spring compliance
d is the damping coefficient
θ_o is the environmental temperature
θ_d is the temperature of the damper 
q is the displacement of the spring
p is the momentum of the mass
s_e is the entropy of the environment
α is the heat transfer coefficient
"""

## give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0, 1.0, 0.0, 0.0]
tspan = (0.0, 20.0)
tsteps = range(tspan[1], tspan[2], length = 1000)
init_params = [1.0; 0.4; 1.0; 2.0; 3.0; 5.0] ### parameters = [m,d,c,θₒ,θd,α]
prob = ODEProblem(ODEfunc_ndho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
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
NN = FastChain(FastDense(4, 40, tanh), ### Multilayer perceptron for the part we don't know
                  FastDense(40, 30, tanh),
                  FastDense(30, 4))
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
      loss = sum(abs2, ode_data .- pred_data) # Just sum of squared error, without mean
      return loss, pred_data
  end

## Callback function to observe training
callback = function(p, loss, pred_data)
    ### plot Ground truth and prediction data
    println(loss)
    x_axis_pred_data = pred_data[1,:]
    y_axis_pred_data = pred_data[2,:]
    z1_axis_pred_data = pred_data[3,:]
    z2_axis_pred_data = pred_data[4,:]
    plt = plot(x_axis_ode_data, y_axis_ode_data, z1_axis_ode_data, label="Ground truth s_d as z")
    plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_ode_data, label="Ground truth s_e as z")
    plt = plot!(x_axis_pred_data, y_axis_pred_data, z1_axis_pred_data, label="Prediction s_d as z")
    plt = plot!(x_axis_pred_data, y_axis_pred_data, z2_axis_pred_data, label="Prediction s_e as z")
    display(plot(plt))
    return false
  end

## prob_neuralode.p are parameters, which are passed into loss_neuralode(p) as its argument
@btime DiffEqFlux.sciml_train(loss_neuralode, neural_params, cb = callback)    



