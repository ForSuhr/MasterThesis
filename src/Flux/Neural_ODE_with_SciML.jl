# https://diffeqflux.sciml.ai/stable/examples/neural_ode_sciml/
## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots

## initial condition
u0 = Float32[2.0; 0.0]
datasize = 30
## set timespan, the first element is the initial time point while the second element is the length of step
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)


"""
ODEFunction f
    f(du,u,p,t): in-place.
    f(u,p,t): returning du. 
"""

## define an ODEFunction, which will generate the data we are trying to fit
function trueODEfunc(du, u, p, t) ### in-place usage: prob_trueode = du
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'  ### Need transposes to make the matrix multiplication work
end

"""
ODEProblem(f::ODEFunction,u0,tspan,p=NullParameters();kwargs...)
    f: The function in the ODE.
    u0: The initial condition.
    tspan: The timespan for the problem.
    p: The parameters.
    kwargs: The keyword arguments passed onto the solves.
"""

## define an ODEProblem with the self-defined ODEFunction above 
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)

## select an algorithm Runge-Kutta method
alg = Tsit5()
## solve the ODEProblem, sol consists of t and u, where t and u are domain and codomain in u(t)
sol = solve(prob_trueode, alg, saveat = tsteps)

## plot u(t)
using Plots
plot(sol)
typeof(sol)

## pick out the data of u in sol into the solution array, where u is a 30-element vector
ode_data = Array(sol)

"""
more examples: https://diffeq.sciml.ai/stable/types/ode_types/#Example-Problems
"""

## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
dudt2 = FastChain((x, p) -> x.^3, # Guess a cubic function
                  FastDense(2, 50, tanh), # Multilayer perceptron for the part we don't know
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps)
### check the parameters prob_neuralode.p in prob_neuralode
prob_neuralode.p

"""
## Chain() from Flux.jl is also available
dudt2 = Chain(x -> x.^3,
              Dense(2, 50, tanh),
              Dense(50, 2))
"""

## Array of predictions from NeuralODE with parameters p starting at initial condition u0
function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
  end

## L2 loss function
function loss_neuralode(p)
      pred = predict_neuralode(p)
      loss = sum(abs2, ode_data .- pred) # Just sum of squared error, without mean
      return loss, pred
  end

## Callback function to observe training
callback = function(p, loss, pred; doplot = true)
    display(loss)
    ### plot current prediction against data
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    if doplot
      display(plot(plt))
    end
    return false
  end

## prob_neuralode.p are parameters, which are passed into loss_neuralode(p) as its argument
result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          cb = callback)

                                          
