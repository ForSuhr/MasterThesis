## using package
using DiffEqFlux, DifferentialEquations, Plots
using Optimization, OptimizationFlux
using BenchmarkTools

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
tspan = (0.0, 9.9)
tsteps = range(tspan[1], tspan[2], length = 100)
init_params = [2.0, 1.0]
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, ImplicitMidpoint(), tstops = tsteps)

## print origin data
ode_data = Array(sol)
q_ode_data = ode_data[1,:]
p_ode_data = ode_data[2,:]
plt = plot(q_ode_data, p_ode_data, label="Ground truth")


NN = Flux.Chain(Flux.Dense(2, 40, tanh),
           Flux.Dense(40, 20, tanh),
          Flux.Dense(20, 2))
prob_neuralode = DiffEqFlux.NeuralODE(NN, tspan, ImplicitMidpoint(), tstops = tsteps)
### check the parameters prob_neuralode.p in prob_neuralode
neural_params = prob_neuralode.p

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end


## L2 loss function
function loss_neuralode(p)
    pred_data = predict_neuralode(p) # solve the Neural ODE with adjoint method
    loss = sum(abs2, ode_data .- pred_data)
    return loss ,pred_data
end


## Callback function to observe training
callback = function(p, loss, pred_data)
    ### plot Ground truth and prediction data
    println(loss)
    if loss > 0.001 
        return false
      else
        return true
      end
end

## first round of training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_neuralode(x), adtype)
optprob1 = Optimization.OptimizationProblem(optf, neural_params)
@time res1 = Optimization.solve(optprob1, ADAM(0.05), callback = callback, maxiters = 100)
## second round of training
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
@time res2 = Optimization.solve(optprob2, ADAM(0.01), callback = callback, maxiters = 300)
## third round of training
optprob3 = Optimization.OptimizationProblem(optf, res2.u)
@time res3 = Optimization.solve(optprob3, ADAM(0.001), callback = callback, maxiters = 300)


## check the trained NN
params_O_NET = res3.u
trajectory_estimate = Array(prob_neuralode(u0, res3.u))
plt = plot(q_ode_data, p_ode_data, label="Ground truth")
plt = plot!(trajectory_estimate[1,:], trajectory_estimate[2,:],  label = "Prediction")
