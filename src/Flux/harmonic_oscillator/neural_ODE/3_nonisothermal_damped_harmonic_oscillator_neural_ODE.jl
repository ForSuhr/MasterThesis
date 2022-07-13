# nonisothermal damped harmonic oscillator ODE
## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots
using BenchmarkTools
## using CUDA

## initial condition
x0 = Float32[1.0;0.0;0.0;0.0]
## set timespan
datasize = 1000
tspan = (1.0f0, 20.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)


## define an ODEFunction, which will generate the data we are trying to fit
function ODEfunc_ndho(df,x,p,t) ### in-place usage
  q = x[1] ### q is the displacement of the spring
  p = x[2] ### p is the momentum of the mass
  s_d = x[3] ### s_d is the entropy of the damper
  s_e = x[4] ### s_e is the entropy of the environment
  d = 0.4  ### d is the damping coefficient
  c = 0.5  ### c is the spring compliance   
  m = 1 ### m is the mass
  θ_d  = 50 ### θ_d is the temperature of the damper 
  θ_o = 20 ### θ is the temperature of the environment
  q_h = 10 ### q_h is the heat flux
  Δθ = abs(θ_d-θ_o) ### Δθ is the temperature difference between damper and environment
  α = q_h/Δθ ### α is the heat transfer coefficient
  v = p/m ### v is the velocity of the mass
  dq = v
  dp = -q/c-d*v 
  ds_d = d*(v^2)/θ_d-α*(θ_d-θ_o)/θ_d
  ds_e = α*(θ_d-θ_o)/θ_o
  df[1] = dq
  df[2] = dp
  df[3] = ds_d
  df[4] = ds_e
end

## define an ODEProblem with the self-defined ODEFunction above 
prob_ODEfunc_ndho = ODEProblem(ODEfunc_ndho, x0, tspan, 0)

## solve the ODEProblem
sol = solve(prob_ODEfunc_ndho, Tsit5(), saveat = tsteps)

## plot sol
plot(sol)
## pick out some data from sol
ode_data = Array(sol)
## plot original data
x_axis_ode_data = ode_data[1,:] ### q
y_axis_ode_data = ode_data[2,:] ### p
z1_axis_ode_data = ode_data[3,:] ### s_d
z2_axis_ode_data = ode_data[4,:] ### s_e
plt = plot(x_axis_ode_data, y_axis_ode_data, z1_axis_ode_data, label="Original s_d as z")
plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_ode_data, label="Original s_e as z")



## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
dudt2 = FastChain(FastDense(4, 40, tanh), ### Multilayer perceptron for the part we don't know
                  FastDense(40, 30, tanh),
                  FastDense(30, 4))
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
    z1_axis_pred_data = pred_data[3,:]
    z2_axis_pred_data = pred_data[4,:]
    plt = plot(x_axis_ode_data, y_axis_ode_data, z1_axis_ode_data, label="Original s_d as z")
    plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_ode_data, label="Original s_e as z")
    plt = plot!(x_axis_pred_data, y_axis_pred_data, z1_axis_pred_data, label="Prediction s_d as z")
    plt = plot!(x_axis_pred_data, y_axis_pred_data, z2_axis_pred_data, label="Prediction s_e as z")
    display(plot(plt))
    return false
  end

## prob_neuralode.p are parameters, which are passed into loss_neuralode(p) as its argument
@btime DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, cb = callback)    



