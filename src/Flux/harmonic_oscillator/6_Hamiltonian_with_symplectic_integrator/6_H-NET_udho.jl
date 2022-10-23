## using package
using DiffEqFlux, DifferentialEquations, Lux
using ReverseDiff, ForwardDiff, FiniteDiff
using Random, Plots
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
tspan = (0.0, 19.9)
tsteps = range(tspan[1], tspan[2], length = 200)
init_params = [2.0, 1.0]
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, ImplicitMidpoint(), tstops = tsteps)

## print origin data
ode_data = Array(sol)
q_ode_data = ode_data[1,:]
p_ode_data = ode_data[2,:]
plt = plot(q_ode_data, p_ode_data, label="Ground truth")


## define neural ODE structure
struct NeuralODE_H_NET{M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:
  Lux.AbstractExplicitContainerLayer{(:model,)}
model::M
solver::So
sensealg::Se
tspan::T
kwargs::K
end

## define neural ODE function
function NeuralODE_H_NET(model::Lux.AbstractExplicitLayer; solver=Midpoint(),
              sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
              tspan=tspan, kwargs...)
return NeuralODE_H_NET(model, solver, sensealg, tspan, kwargs)
end

## replace the RHS with neural network
function (n::NeuralODE_H_NET)(x, ps, st) 
  function dudt(u, p, t)
    #du, _ = n.model(u, p, st)
    du = vec(hamiltonian_forward(n.model, p, st, u))
  end
  prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
  return solve(prob, n.solver; sensealg=n.sensealg, saveat = tsteps), st
end


H_NET = Lux.Chain(Lux.Dense(2, 40, tanh),
                  Lux.Dense(40, 20, tanh),
                  Lux.Dense(20, 1))
#prob_neuralode = NeuralODE_H_NET(H_NET, solver=ImplicitMidpoint(), tspan=tspan)
prob_neuralode = NeuralODE_H_NET(H_NET, solver=Midpoint(), tspan=tspan)

# initial random parameters and states
rng = Random.default_rng()
neural_params, state = Lux.setup(rng, H_NET)


function hamiltonian_forward(NN, ps, st, x)
  #H = Flux.gradient(x -> sum(NN(x, ps, st)[1]), x)[1]
  #H = ReverseDiff.gradient(x -> sum(NN(x, ps, st)[1]), x)
  #H = ForwardDiff.gradient(x -> sum(NN(x, ps, st)[1]), x)  
  H = FiniteDiff.finite_difference_gradient(x -> sum(NN(x, ps, st)[1]), x)
  n = size(x, 1) ÷ 2
  return cat(H[(n + 1):2n, :], -H[1:n, :], dims=1)
end

output = prob_neuralode(u0, neural_params, state)[1]
plot(output)
Array(output)

#AutoForwardDiff
H = ForwardDiff.gradient(x -> sum(H_NET(x, neural_params, state)[1]), u0)
# symplectic gradient of H_NET
X_H = vec(hamiltonian_forward(H_NET, neural_params, state, u0))
output = prob_neuralode(X_H, neural_params, state)[1]
plot(output)

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
  pred_data, _ = prob_neuralode(u0, p, state)
  return Array(pred_data)
end


## L2 loss function
function loss_neuralode(p)
    pred_data = predict_neuralode(p) # solve the Neural ODE with adjoint method
    loss = sum(abs2, ode_data .- pred_data)
    return loss ,Array(pred_data)
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
#adtype = Optimization.AutoZygote()
#adtype = Optimization.AutoReverseDiff()
#adtype = Optimization.AutoForwardDiff()
adtype = Optimization.AutoFiniteDiff()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob1 = Optimization.OptimizationProblem(optf, Lux.ComponentArray(neural_params))
@time res1 = Optimization.solve(optprob1, ADAM(0.01), callback = callback, maxiters = 10)
## second round of training
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
@time res2 = Optimization.solve(optprob2, ADAM(0.001), callback = callback, maxiters = 10)
## third round of training
optprob3 = Optimization.OptimizationProblem(optf, res2.u)
@time res3 = Optimization.solve(optprob3, ADAM(0.0005), callback = callback, maxiters = 10)
params_H_NET = res3.u

# repeat training
optprob4 = Optimization.OptimizationProblem(optf, params_H_NET)
@time res = Optimization.solve(optprob4, ADAM(0.00001), callback = callback, maxiters = 100)
params_H_NET = res.u

## check the trained NN
trajectory_estimate = Array(prob_neuralode(u0, params_H_NET, state)[1])
plt = plot(q_ode_data, p_ode_data, label="Ground truth")
plt = plot!(trajectory_estimate[1,:], trajectory_estimate[2,:],  label = "Prediction")

## save the parameters
using JLD
save(joinpath(@__DIR__, "results", "params_H_NET.jld"), "params_H_NET", params_H_NET)
params_H_NET = load(joinpath(@__DIR__, "results", "params_H_NET.jld"), "params_H_NET")
