# https://diffeqflux.sciml.ai/stable/examples/tensor_layer/

## use packages
using DiffEqFlux, DifferentialEquations, LinearAlgebra
using Optimization, OptimizationFlux
using Plots

function ODEfunc_ndho(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
    q, p, sd, sₑ = u
    m, d, c, θₒ, θd, α = params
    ## ODEs
    du[1] = p/m
    du[2] = -q/c-d*p/m
    du[3] = d*((p/m)^2)/θd-α*(θd-θₒ)/θd
    du[4] = α*(θd-θₒ)/θₒ
end

u0 = Float32[1.0,1.0,0.0,0.0]
tspan = (0.0,10.0)
tsteps = collect(0.0:0.1:tspan[2])
init_params = Float32[1.0; 0.4; 1.0; 2.0; 3.0; 5.0]
prob = ODEProblem{true}(ODEfunc_ndho, u0, tspan, init_params)
ode_data = Array(solve(prob,Tsit5(),saveat=tsteps))

## construct legendre basis
A = [LegendreBasis(10), LegendreBasis(10)]
## construct neural network with tensor layers
nn = TensorLayer(A, 1)
f = x -> min(30one(x),x)

params_PIML = zeros(104)

function ODEfunc_ndho_pred(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
    q, p, sd, sₑ = u
    m, d, c, θₒ, θd, α = init_params
    ## ODEs
    du[1] = p/m
    du[2] = -q/c-d*p/m
    du[3] = d*((p/m)^2)/θd-f(nn(u,params[5:end])[1])
    du[4] = f(nn(u,params[5:end])[1])
end


prob_pred = ODEProblem{true}(ODEfunc_ndho_pred,u0,tspan,init_params)

function predict_adjoint(params)
    x = Array(solve(prob_pred,Tsit5(),p=params,saveat=tsteps,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end
  
function loss_adjoint(init_params)
  pred_data = predict_adjoint(init_params)
  # rss = sum(abs2, ode_data .- pred_data)
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

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params_PIML)
res1 = Optimization.solve(optprob, ADAM(0.05), callback = callback, maxiters = 100)


optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, ADAM(0.01), callback = callback,maxiters = 100)
opt = res2.u


ode_data
data_pred = predict_adjoint(res2.u)
plt = plot(ode_data[1,:], ode_data[2,:], ode_data[3,:], label = "Ground truth")
plt = plot!(data_pred[1,:], data_pred[2,:], data_pred[3,:], label = "Prediction")