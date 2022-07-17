# https://diffeqflux.sciml.ai/stable/examples/tensor_layer/

##
using DiffEqFlux, Optimization, OptimizationFlux, DifferentialEquations, LinearAlgebra
# k, α, β, γ = 1, 0.1, 0.2, 0.3


function dxdt_train(du,u,p,t)
  du[1] = u[2]
  du[2] = -k*u[1] - α*u[1]^3 - β*u[2] - γ*u[2]^3
end

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
init_params_ode = Float32[1.0; 0.4; 1.0; 2.0; 3.0; 5.0]
prob_train = ODEProblem{true}(ODEfunc_ndho,u0,tspan,init_params_ode)
data_train = Array(solve(prob_train,Tsit5(),saveat=tsteps))

A = [LegendreBasis(10), LegendreBasis(10)]
nn = TensorLayer(A, 1)

f = x -> min(30one(x),x)

function dxdt_pred(du,u,p,t)
  du[1] = u[2]
  du[2] = -p[1]*u[1] - p[2]*u[2] + f(nn(u,p[3:end])[1])
end

function ODEfunc_ndho_pred(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
    q, p, sd, sₑ = u
    m, d, c, θₒ, θd, α = init_params
    ## ODEs
    du[1] = p/m
    du[2] = -q/c-d*p/m
    # du[3] = d*((p/m)^2)/θd-α*(θd-θₒ)/θd
    du[3] = d*((p/m)^2)/θd-f(nn(u,params[5:end])[1])
    du[4] = f(nn(u,params[5:end])[1])
end

# given that we don't know m, so params=[d,θₒ,c]

prob_pred = ODEProblem{true}(ODEfunc_ndho_pred,u0,tspan,init_params)

function predict_adjoint(init_params)
    x = Array(solve(prob_pred,Tsit5(),p=init_params,saveat=tsteps,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
  end
  
function loss_adjoint(init_params)
    x = predict_adjoint(init_params)
    loss = sum(norm.(x - data_train))
    return loss
end
  
  iter = 0
function callback(θ,l)
  global iter
  iter += 1
  if iter%10 == 0
     println(l)
  end
  return false
end

params_PIML = zeros(104)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params_PIML)
res1 = Optimization.solve(optprob, ADAM(0.05), callback = callback, maxiters = 150)

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, ADAM(0.001), callback = callback,maxiters = 150)
opt = res2.u

using Plots
data_pred = predict_adjoint(res1.u)
plot(tsteps, data_train[1,:], label = "X (ODE)")
plot!(tsteps, data_train[2,:], label = "V (ODE)")
plot!(tsteps, data_pred[1,:], label = "X (NN)")
plot!(tsteps, data_pred[2,:],label = "V (NN)")