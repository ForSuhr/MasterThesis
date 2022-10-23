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
tspan = (0.0, 9.9)
tsteps = range(tspan[1], tspan[2], length = 100)
init_params = [2.0, 1.0]
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, ImplicitMidpoint(), tstops = tsteps)

## print origin data
ode_data = Array(sol)
plot(sol)
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
function NeuralODE_H_NET(model::Lux.AbstractExplicitLayer; solver=ImplicitMidpoint(),
              sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
              tspan=tspan, kwargs...)
return NeuralODE_H_NET(model, solver, sensealg, tspan, kwargs)
end


## replace the RHS with neural network
function (n::NeuralODE_H_NET)(x, ps, st) 
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    return solve(prob, n.solver; sensealg=n.sensealg, tstops = tsteps), st
end



H_NET = Lux.Chain(Lux.Dense(2, 40, tanh),
                  Lux.Dense(40, 20, tanh),
                  Lux.Dense(20, 1))
prob_neuralode = NeuralODE_H_NET(H_NET, solver=ImplicitMidpoint(), tspan=tspan)
# initial random parameters and states
rng = Random.default_rng()
neural_params, state = Lux.setup(rng, H_NET)

function dudt(u, ps, t)
    q, p = u
    m, c = init_params
    # du = u
    du = [p/m, hamiltonian_forward(H_NET, ps, state, u)[2]]
    # du[1] = hamiltonian_forward(H_NET, ps, state, u)[1]
    # du[2] = hamiltonian_forward(H_NET, ps, state, u)[2]
    # du[1] = p/m
    # du[2] = hamiltonian_forward(H_NET, ps, state, u)[2]
    return du
end

# dudt(u0, neural_params, tspan)


function hamiltonian_forward(NN, ps, st, x)
    #H = Flux.gradient(x -> sum(NN(x, ps, st)[1]), x)[1]
    #H = ReverseDiff.gradient(x -> sum(NN(x, ps, st)[1]), x)
    #H = ForwardDiff.gradient(x -> sum(NN(x, ps, st)[1]), x) 
    ∇H = FiniteDiff.finite_difference_gradient(x -> sum(NN(x, ps, st)[1]), x)
    return return cat(∇H[2, :], -∇H[1, :], dims=1)
end

∇H = FiniteDiff.finite_difference_gradient(x -> sum(H_NET(x, neural_params, state)[1]), u0)
# symplectic gradient of H_NET
X_H = hamiltonian_forward(H_NET, neural_params, state, u0)[2]


output = prob_neuralode(u0, neural_params, state)[1]
plot(output)
Array(prob_neuralode(u0, neural_params, state)[1])


## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
  pred_data = Array(prob_neuralode(u0, p, state)[1])
  return pred_data
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
@time res1 = Optimization.solve(optprob1, ADAM(0.001), callback = callback, maxiters = 10)
## second round of training
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
@time res2 = Optimization.solve(optprob2, ADAM(0.0005), callback = callback, maxiters = 300)
## third round of training
optprob3 = Optimization.OptimizationProblem(optf, res2.u)
@time res3 = Optimization.solve(optprob3, ADAM(0.0001), callback = callback, maxiters = 1000)


## check the trained NN
params_structured_N_NET = res3.u
plot(prob_neuralode(u0, params_structured_N_NET, state)[1])
trajectory_estimate = Array(prob_neuralode(u0, params_structured_N_NET, state)[1])
plt = plot(q_ode_data, p_ode_data, label="Ground truth")
plt = plot!(trajectory_estimate[1,:], trajectory_estimate[2,:],  label = "Prediction")
