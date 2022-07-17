# undamped harmonic oscillator ODE

## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots
using BenchmarkTools
using ForwardDiff, Zygote

## initial condition
u0 = Float32[0.0, 1.0]
## set timespan
tspan = (1.0f0, 20.0f0)
datasize = 100
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
## plot Ground truth data (q,p)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
# plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth")


## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
NN = FastChain(FastDense(2, 10, tanh), ### Multilayer perceptron for the part we don't know
               #FastDense(40, 40, tanh),
               FastDense(10, 2))


## set neural ode problem
prob_neuralode = NeuralODE(NN, tspan, Tsit5(), saveat = tsteps)
## initial parameters for neural network
neural_params = prob_neuralode.p  ### equivalent to initial_params(NN)
initial_params(NN) 
prob_neuralode_multiple_shooting = ODEProblem((u,p,t)->NN(u,p), u0, tspan, neural_params)

# Define parameters for Multiple Shooting
group_size = 40
continuity_term = 200

function loss_function(ode_data, pred_data)
  x_axis_ode_data = ode_data[1,:]
  y_axis_ode_data = ode_data[2,:]
  error = sum(abs2, ode_data .- pred_data)
	return error
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, ode_data, tsteps, prob_neuralode_multiple_shooting, loss_function, Tsit5(),
                          group_size; continuity_term)
end

  
function plot_multiple_shoot(plt, preds, group_size)
	step = group_size-1
	ranges = DiffEqFlux.group_ranges(datasize, group_size)
	### plot the original data
  plt = scatter(x_axis_ode_data, y_axis_ode_data, label = "Ground truth")
  for (i, rg) in enumerate(ranges)
    plt = plot!(preds[i][1,:], preds[i][2,:], markershape=:circle, label="Prediction")
  end
  # print(pred_data_array)

	# plot!(plt, step_ode_data[:,i], preds[i][2,:], markershape=:circle, label="Prediction")
  #frame(anim)
  display(plot(plt))
end

#anim = Animation()
callback = function (p, loss, preds; doplot = true)
  display(loss)
  if doplot
    plot_multiple_shoot(plt, preds, group_size)
  end
  return false
end

## prob_neuralode.p are parameters, which are passed into loss_neuralode(p) as its argument
## optimizer chain of ADAM -> BFGS is used by default
@btime DiffEqFlux.sciml_train(loss_multiple_shooting, prob_neuralode.p, cb = callback)
