## using package
using DiffEqFlux, DifferentialEquations, Plots
using Optimization, OptimizationFlux
using BenchmarkTools
using Noise

function ODEfunc_idho(du,u,params,t) ### du=[̇q,̇p,̇sₑ], u=[q,p,sₑ], params=[m,d,θₒ,c]
  q, p = u
  m, c, d, θₒ = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c-d*p/m
  #du[3] = d*(p/m)^2/θₒ
end

# give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0, 1.0] # q, p
tspan = (0.0, 49.9)
tsteps = range(tspan[1], tspan[2], length = 500)
init_params = [2.0, 1.0, 1.0, 20.0] # parameters = [m, c, d, θₒ]
prob = ODEProblem(ODEfunc_idho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
#sol = solve(prob, ImplicitMidpoint(), tstops = tsteps)
sol = solve(prob, Midpoint(), saveat = tsteps)

## print origin data
ode_data = Array(sol)
# ode_data = add_gauss(ode_data, 0.01)
q_ode_data = ode_data[1,:]
p_ode_data = ode_data[2,:]
plt = plot(q_ode_data, p_ode_data, label="Ground truth")


NN = Chain(Dense(1, 40, tanh),
            Dense(40, 40, tanh),
            Dense(40, 1))
p, re = Flux.destructure(NN)
neural_params = p
# The model weights are destructured into a vector of parameters
size_neural_params = length(neural_params)
zeros_params = zeros(size_neural_params)
## the first output of the NN
re(zeros_params)([u0[1]])
re(neural_params)([u0[1]])[1]

function Structured_O_NET(du,u,ps,t) ### params = params_PIML
    ## conversion
    q, p = u
    m, c = init_params
    ## ODEs
    du[1] = p/m
    du[2] = -q/c + re(ps)([p])[1]
    #du[2] = NN(q, ps[1:size_neural_params])[1]
end


prob_pred = ODEProblem(Structured_O_NET, u0, tspan, init_params)

## Array of predictions from NeuralODE with parameters p starting at initial condition x0
function predict_neuralode(p)
    # Array(solve(prob_pred, ImplicitMidpoint(), p=p, tstops = tsteps,
    # sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
    Array(solve(prob_pred, Midpoint(), p=p, saveat = tsteps,
    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
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
@time res1 = Optimization.solve(optprob1, ADAM(0.002), callback = callback, maxiters = 20)
## second round of training
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
@time res2 = Optimization.solve(optprob2, ADAM(0.0005), callback = callback, maxiters = 20)
## third round of training
optprob3 = Optimization.OptimizationProblem(optf, res2.u)
@time res3 = Optimization.solve(optprob3, ADAM(0.0001), callback = callback, maxiters = 100)
params_structured_O_NET = res3.u

## check phase portrait
trajectory_estimate_structured_O_NET = Array(solve(prob_pred, ImplicitMidpoint(), p=params_structured_O_NET, tstops = tsteps,
sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
plt = plot(q_ode_data, p_ode_data, xlims=(-2,2), ylims=(-2,2), lw=4, label="Ground truth", linestyle=:solid)
plt = plot!(trajectory_estimate_structured_O_NET[1,:], trajectory_estimate_structured_O_NET[2,:], lw=4,  label = "Structured O-NET", linestyle=:dot)

# Calculate training loss
training_error_O_NET = sum((ode_data .- trajectory_estimate_structured_O_NET).^2)/2/500
begin
min_error = (findmax(sum((ode_data[:, 1:25] .- trajectory_estimate_structured_O_NET[:, 1:25]).^2, dims=1)/2)[1] +
  findmin(sum((ode_data[:, 1:25] .- trajectory_estimate_structured_O_NET[:, 1:25]).^2, dims=1)/2)[1])/2
max_error = (findmax(sum((ode_data[:, :] .- trajectory_estimate_structured_O_NET[:, :]).^2, dims=1)/2)[1] +
  findmin(sum((ode_data[:, :] .- trajectory_estimate_structured_O_NET[:, :]).^2, dims=1)/2)[1])/2
end
(min_error + max_error)/2
max_error - (min_error + max_error) /2
# Calculate test loss
test_tspan = (50.0, 99.9)
test_tsteps = range(test_tspan[1], test_tspan[2], length = 500)
test_prob = ODEProblem(ODEfunc_idho, u0, test_tspan, init_params)
test_sol = solve(test_prob, ImplicitMidpoint(), tstops = test_tsteps)
test_data = Array(test_sol)
test_data = add_gauss(test_data, 0.01)
q_test_data = test_data[1,:]
p_test_data = test_data[2,:]
test_error_structured_O_NET = sum((test_data .- trajectory_estimate_structured_O_NET).^2)/2/200
test_prob_pred = ODEProblem(Structured_O_NET, u0, test_tspan, init_params)
test_trajectory_estimate_structured_O_NET = Array(solve(test_prob_pred, ImplicitMidpoint(), p=params_structured_O_NET, tstops = test_tsteps,
sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
test_error_O_NET = sum((test_data .- test_trajectory_estimate_structured_O_NET).^2)/2/500
begin
  min_error = (findmax(sum((test_data[:, 1:25] .- test_trajectory_estimate_structured_O_NET[:, 1:25]).^2, dims=1)/2)[1] +
              findmin(sum((test_data[:, 1:25] .- test_trajectory_estimate_structured_O_NET[:, 1:25]).^2, dims=1)/2)[1])/2
  max_error = (findmax(sum((test_data[:, :] .- test_trajectory_estimate_structured_O_NET[:, :]).^2, dims=1)/2)[1] +
              findmin(sum((test_data[:, :] .- test_trajectory_estimate_structured_O_NET[:, :]).^2, dims=1)/2)[1])/2
  end
(min_error + max_error)/2
max_error - (min_error + max_error) /2
  
# check L2 test error
l2_error_structured_O_NET = vec(sum((test_data .- trajectory_estimate_structured_O_NET).^2, dims=1)/2)
plt = plot!(tsteps, l2_error_structured_O_NET, lw=3, label="Structured O-NET")

# check Hamiltonian
H_ground_truth = ode_data[2,:].^2/(2*init_params[1]) + ode_data[1,:].^2/(2*init_params[2])
H_structured_O_NET = trajectory_estimate_structured_O_NET[2,:].^2/(2*init_params[1]) + trajectory_estimate_structured_O_NET[1,:].^2/(2*init_params[2])
plt = plot(tsteps, H_ground_truth, lw=3, label="Ground truth")
plt = plot!(tsteps, H_structured_O_NET, lw=3, label="Structured O-NET", linestyle=:dot)
(- findmin(H_structured_O_NET - H_ground_truth)[1] + findmax(H_structured_O_NET - H_ground_truth)[1])/2
test_H_ground_truth = test_data[2,:].^2/(2*init_params[1]) + test_data[1,:].^2/(2*init_params[2])
test_H_structured_O_NET = test_trajectory_estimate_structured_O_NET[2,:].^2/(2*init_params[1]) + test_trajectory_estimate_structured_O_NET[1,:].^2/(2*init_params[2])
plt = plot(test_tsteps, test_H_ground_truth, lw=3, label="Ground truth")
plt = plot!(test_tsteps, test_H_structured_O_NET, lw=3, label="O-NET", ylabel="Hamiltonian", xlabel="Time step", linestyle=:dot)
(- findmin(test_H_structured_O_NET - test_H_ground_truth)[1] + findmax(test_H_structured_O_NET - test_H_ground_truth)[1])/2
(findmin(test_H_structured_O_NET)[1] + findmax(test_H_structured_O_NET)[1])/2
