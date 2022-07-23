# undamped harmonic oscillator ODE
## using package
using DiffEqFlux, DifferentialEquations, Plots
using Optimization, OptimizationFlux
using BenchmarkTools
using Zygote

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
tspan = (0.0, 20.0)
tsteps = range(tspan[1], tspan[2], length = 100)
init_params = [1.5, 1.0]
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, Tsit5(), saveat = tsteps)

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth")

## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
NN = FastChain(FastDense(2, 20, tanh), ### Multilayer perceptron for the part we don't know
                  FastDense(20, 10, tanh),
                  FastDense(10, 2))
prob_neuralode = NeuralODE(NN, tspan, Tsit5(), saveat = tsteps)
### check the parameters prob_neuralode.p in prob_neuralode
neural_params = prob_neuralode.p
initial_params(NN)
dNN_dq = Zygote.gradient(u -> NN(u, initial_params(NN))[1], u0[1])[1] ## ∂H/∂q
dNN_dp = Zygote.gradient(u -> NN(u, initial_params(NN))[2], u0[2])[1] ## ∂H/∂p
NN(u0, neural_params)



## define ODEs
function ODEfunc_udho_HNN(du,u,params,t)
    ## ODEs
    du[1] = dNN_dq = Zygote.gradient(u -> NN(u, params)[1], u0[1])[1] ## ∂H/∂q
    du[2] = dNN_dp = Zygote.gradient(u -> NN(u, params)[2], u0[2])[1] ## ∂H/∂p
  end

prob_HNN = ODEProblem(ODEfunc_udho_HNN, u0, tspan)


sol_HNN = solve(prob_HNN, Tsit5(), saveat = tsteps, initial_params(NN))
hnn_data = Array(sol_HNN)
x_axis_HNN_data = hnn_data[1,:]
y_axis_HNN_data = hnn_data[2,:]
plt = plot!(x_axis_HNN_data, y_axis_HNN_data, label="Predition")



## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
    Array(solve(prob_HNN, Tsit5(), saveat = tsteps))
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
    plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth")
    plot!(plt,x_axis_pred_data, y_axis_pred_data, label = "Prediction")
    display(plot(plt))
    if loss > 20 
      return false
    else
      return true
    end
end


adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_neuralode(x), adtype)
optprob1 = Optimization.OptimizationProblem(optf, neural_params)
res1 = Optimization.solve(optprob1, ADAM(0.05), callback = callback, maxiters = 100)

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, ADAM(0.01), callback = callback, maxiters = 1000)

optprob3 = Optimization.OptimizationProblem(optf, res2.u)
res3 = Optimization.solve(optprob3, ADAM(0.001), callback = callback, maxiters = 1000)

## check the trained NN
NN(u0, res2.u)