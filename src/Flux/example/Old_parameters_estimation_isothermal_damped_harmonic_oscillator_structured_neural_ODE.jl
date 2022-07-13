# isothermal damped harmonic oscillator ODE
## using package
using Revise
using DiffEqFlux, DifferentialEquations, Plots
## using ModelingToolkit
using DiffEqParamEstim
using RecursiveArrayTools ### for VectorOfArray
using LeastSquaresOptim ### for LeastSquaresProblem
using Statistics ### for using mean()

## define a structured neural ODE
"""
m is the mass
c is the spring compliance
d is the damping coefficient
θ_o is the environmental temperature
q is the displacement of the spring
p is the momentum of the mass
s_e is the entropy of the environment

## using ModelingToolkit
function structured_neural_ODE()
  @parameters m d θ_o c
  @variables t q(t) p(t) s_e(t) 
  D = Differential(t)
  ### DifferentialEquations
  eqs = [D(q) ~ p/m,
         D(p) ~ -q/c-d*p/m,
         D(s_e) ~ d*((p/m)^2)/θ_o]
  @named DiffEq = ODESystem(eqs,t,[q,p,s_e,c],[m,d,θ_o])
  return DiffEq
end
"""
function structured_neural_ODE(du,u,p,t) ### du=[̇q,̇p,̇sₑ], u=[q,p,sₑ], p=[m,d,θₒ,c]
  du[1] = u[2]/p[1]
  du[2] = -u[1]/p[4]-p[2]*u[2]/p[1]
  du[3] = p[2]*((u[2]/p[1])^2)/p[3]
end

## set initial condition
u0 = Float32[1.0; 1.0; 0.0] ### u0 = [q,p,sₑ]
## set timespan
datasize = 100
tspan = (0.0f0, 10.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
## set an initial set of parameters
parameters_initial = [1.0; 0.4; 20; 1.0] ### parameters = [m,d,θₒ,c]

## solve the ODEProblem
prob_ODEfunc_idho_initial = ODEProblem(structured_neural_ODE, u0, tspan, parameters_initial)
sol_initial = solve(prob_ODEfunc_idho_initial, Tsit5(), saveat = tsteps)
### plt_sol = plot(sol_initial)
## pick out some data from sol
ode_data = Array(sol_initial)
## plot original data (q,p)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
z_axis_ode_data = ode_data[3,:]
plt_trajectory = plot(x_axis_ode_data, y_axis_ode_data, z_axis_ode_data, label="Original")





#######################################################

## build a LeastSquaresOptim object so that we can solve it by using LeastSquaresOptim.jl in the following
loss_function = build_lsoptim_objective(prob_ODEfunc_idho_initial,tsteps,ode_data,Tsit5())
## using LeastSquaresOptim.optimize!
### https://github.com/matthieugomez/LeastSquaresOptim.jl
"""
x is an initial set of parameters.
f!(out, x) that writes f(x) in out.
the option output_length to specify the length of the output vector.
Optionally, g! a function such that g!(out, x) writes the jacobian at x in out. Otherwise, the jacobian will be computed following the :autodiff argument.
"""
x = [2.0; 1.4; 10.0; 2.0]
res = optimize!(LeastSquaresProblem(x = x, f! = loss_function,
                output_length = length(tsteps)*length(prob_ODEfunc_idho_initial.u0)),
                LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR()))

println(res.minimizer)
parameters_predicted = res.minimizer

## solve the ODEProblem
prob_ODEfunc_idho_predicted = ODEProblem(structured_neural_ODE, u0, tspan, parameters_predicted)
sol_predicted = solve(prob_ODEfunc_idho_predicted, Tsit5(), saveat = tsteps)
### plt_sol = plot(sol_predicted)
## pick out some data from sol
predict_data = Array(sol_predicted)
## plot original data (q,p)
x_axis_predict_data = predict_data[1,:]
y_axis_predict_data = predict_data[2,:]
z_axis_predict_data = predict_data[3,:]
plt_trajectory = plot!(x_axis_predict_data, y_axis_predict_data, z_axis_predict_data, label="Predicted")


#######################################################
"""
loss_function = build_loss_objective(prob_ODEfunc_idho_initial,Tsit5(),L2Loss(tsteps,ode_data),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(loss_function, x, Optim.BFGS())
"""
#######################################################


"""
## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
NN = FastChain(FastDense(3, 20, tanh), ### Multilayer perceptron for the part we don't know
                  FastDense(20, 10, tanh),
                  FastDense(10, 3))
prob_neuralode = NeuralODE(NN, tspan, Tsit5(), saveat = tsteps)
### inspect the parameters prob_neuralode.p in prob_neuralode
prob_neuralode.p

## Array of predictions from NeuralODE with parameters p starting at initial condition u0
function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
  end

## L2 loss function
function loss_neuralode(p)
      pred_data = predict_neuralode(p)
      loss = mean(sum(abs2, ode_data .- pred_data)) ### mean squared error
      return loss, pred_data
  end

## Callback function to observe training
callback = function(p, loss, pred_data)
    ### plot original and prediction data
    println(loss)
    x_axis_pred_data = pred_data[1,:]
    y_axis_pred_data = pred_data[2,:]
    z_axis_pred_data = pred_data[3,:]
    plt = plot(x_axis_ode_data, y_axis_ode_data, z_axis_ode_data, label="Original")
    plt = plot!(x_axis_pred_data, y_axis_pred_data, z_axis_pred_data, label = "Prediction")
    display(plot(plt))
    return false
  end

## prob_neuralode.p are parameters, which are passed into loss_neuralode(p) as its argument
result = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, cb = callback)
"""
