using Lux
using Random
using OrdinaryDiffEq
using DiffEqSensitivity
using Optimization
using OptimizationFlux
using Plots

## ODE function of undamped harmonic oscillator
function ODEfunc_udho(du,u,params,t)
    ## conversion
    q, p = u
    m, c = params
    ## ODEs
    du[1] = p/m
    du[2] = -q/c
end

## initial conditions, timesteps, parameters
u0 = [1.0; 1.0]
tspan = (0.0, 20.0)
datasize = 100
tsteps = collect(range(tspan[1], tspan[2], length = datasize))
init_params = [1.5, 1.0]

## construct a ODE problem
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem, the solution is the trajectories
sol = solve(prob, Tsit5(), saveat = tsteps)

## print origin data in form of phase portrait
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth")

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
function (n::NeuralODE)(x, ps, st) 
 function dudt(u, p, t)
     u_, _ = n.model(u, p, st)
     return u_
 end
 prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps) ## "false" means out-of-place usage, as we don't have real dudt in neural ODE, so the out-of-place usage is neccessary.
 return solve(prob, n.solver; sensealg=n.sensealg, saveat = tsteps), st
end

## collect layers
NN = Lux.Chain(Lux.Dense(2, 20, tanh), 
               Lux.Dense(20, 10, tanh),
               Lux.Dense(10, 2))

## construct an neural ODE problem
prob_neuralode = NeuralODE(NN, solver=Tsit5(), tspan=tspan)

## initial random parameters and states
rng = Random.default_rng()
ps, st = Lux.setup(rng, NN)

## solve the neural ODE problem
function predict_neuralode(p)
  pred_data = Array(prob_neuralode(u0, p, st)[1])
  return pred_data
end

## loss function
function loss_neuralode(p)
    pred_data = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred_data) # Just sum of squared error, without mean
    return loss, pred_data
end

## Callback function for observing training
callback = function(params, loss, pred_data)
    ### plot Ground truth and prediction data
    println(loss)
    x_axis_pred_data = pred_data[1,:]
    y_axis_pred_data = pred_data[2,:]
    plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth", xlabel="q", ylabel="p")
    plot!(plt,x_axis_pred_data, y_axis_pred_data, label = "Prediction", xlabel="q", ylabel="p")
    """
    q = pred_data[1,:]
    p = pred_data[2,:]
    m, c = init_params
    H = p.^2/(2m) + q.^2/(2c)
    plt = plot(tsteps, q, label="Position")
    plt = plot!(tsteps, p, label="Momentum")
    plt = plot!(tsteps, H, label="Hamiltonian")
    """
    display(plot(plt))
    if loss > 0.1 
      return false
    else
      return true
    end
  end

## first round of training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_neuralode(x), adtype)
optprob1 = Optimization.OptimizationProblem(optf, Lux.ComponentArray(ps))
res1 = Optimization.solve(optprob1, ADAM(0.05), callback = callback, maxiters = 100)
## second round of training
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, ADAM(0.01), callback = callback, maxiters = 300)
## third round of training
optprob3 = Optimization.OptimizationProblem(optf, res2.u)
res3 = Optimization.solve(optprob3, ADAM(0.001), callback = callback, maxiters = 500)

## inspect the trained model
out_NN, _ = NN(u0, res3.u, st)
out_NN