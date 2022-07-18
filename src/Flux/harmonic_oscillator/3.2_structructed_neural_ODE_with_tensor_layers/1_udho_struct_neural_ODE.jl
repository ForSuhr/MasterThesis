# https://diffeqflux.sciml.ai/stable/examples/tensor_layer/

## use packages
using DiffEqFlux, DifferentialEquations, LinearAlgebra
using Optimization, OptimizationFlux
using Plots

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
tsteps = collect(range(tspan[1], tspan[2], length = 100))
init_params = [1.5, 1.0]
prob = ODEProblem{true}(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem
sol = solve(prob, Tsit5(), saveat = tsteps)

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth")

## construct legendre basis
A = [LegendreBasis(10), LegendreBasis(10)] ### need 10*10 parameters
## construct neural network with tensor layers
nn = TensorLayer(A, 1)
f = x -> min(30one(x),x)

## replace a part of the ODEs with a neural network
function ODEfunc_udho_pred(du,u,params,t) ### params = params_PIML
  ## conversion
  q, p = u
  m, c = init_params
  ## ODEs
  du[1] = p/m
  du[2] = f(nn(u,params[2:end])[1])
end

## initial parameters for training
params_PIML = zeros(101)
## construct an ODE problem with NN replacement part
prob_pred = ODEProblem{true}(ODEfunc_udho_pred, u0, tspan, init_params)

function predict_adjoint(params)
    x = Array(solve(prob_pred, Tsit5(), p=params, saveat=tsteps,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end
  
function loss_adjoint(init_params)
  pred_data = predict_adjoint(init_params)
  error = sum(norm.(ode_data - pred_data))
  return error
end

iter = 0
function callback(params, loss)
  global iter
  iter += 1
  if iter%10 == 0
     println(loss)
  end
  return false
end

## training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params_PIML)
res1 = Optimization.solve(optprob, ADAM(0.05), callback = callback, maxiters = 150)

"""
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, ADAM(0.001), callback = callback,maxiters = 150)
opt = res2.u
"""

data_pred = predict_adjoint(res1.u)
plot(tsteps, ode_data[1,:], label = "X (ODE)")
plot!(tsteps, ode_data[2,:], label = "V (ODE)")
plot!(tsteps, data_pred[1,:], label = "X (NN)")
plot!(tsteps, data_pred[2,:],label = "V (NN)")