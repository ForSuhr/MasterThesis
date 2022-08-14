using OrdinaryDiffEq
using Lux
using Random
using DiffEqSensitivity
using BenchmarkTools


function timesteps(start::Float64, step::Float64, stop::Float64)
    tspan = (start, stop)
    datasize = Int64.((stop-start)/step)
    tsteps = collect(range(tspan[1], tspan[2], length=datasize))
    return tspan, tsteps
end


function ODE_integrator(fun, u0, timesteps, params, alg=Tsit5())
    tspan, tsteps = timesteps
    prob = ODEProblem(fun, u0, tspan, params)
    sol = solve(prob, alg, saveat = tsteps)
    return Array(sol)
end


## define neural ODE structure
struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:
    Lux.AbstractExplicitContainerLayer{(:model,)}
 model::M
 solver::So
 sensealg::Se
 tspan::T
 kwargs::K
end


## define neural ODE function
function NeuralODE(model::Lux.AbstractExplicitLayer; solver=Tsit5(),
                sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
                tspan=tspan, kwargs...)
 return NeuralODE(model, solver, sensealg, tspan, kwargs)
end


## replace the RHS with neural network
function (n::NeuralODE)(x, ps, st, tsteps) 
 function dudt(u, p, t)
     du, _ = n.model(u, p, st)
     return du
 end
 prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
 return solve(prob, n.solver; sensealg=n.sensealg, saveat = tsteps), st
end


function neural_network(input, hidden, output; activation_function=tanh)
    NN = Lux.Chain(Lux.Dense(input, hidden, activation_function), 
                Lux.Dense(hidden, hidden, activation_function),
                Lux.Dense(hidden, output))
    return NN
end


function params_initial(NN)
    ## initial random parameters and states
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, NN)
    return ps, st
end


function neural_ODE_integrator(NN, u0, timesteps, params, state, alg=Tsit5())
    tspan, tsteps = timesteps
    ## construct an neural ODE problem
    prob_neuralode = NeuralODE(NN, solver=alg, tspan=tspan)
    pred_data = Array(prob_neuralode(u0, params, state, tsteps)[1])
    return pred_data
end


function decompose_ps(neural_params)
    ps, _ = neural_params
    neural_initial_params = Lux.ComponentArray(ps)
    return neural_initial_params
end

function decompose_st(neural_params)
    _, st = neural_params
    return st
end


