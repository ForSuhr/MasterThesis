using DiffEqFlux, DifferentialEquations, Statistics, Plots
using Noise
using ReverseDiff

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
u0 = [1, 1]
tspan = (0.0f0, 49.0f0)
tsteps = range(tspan[1], tspan[2], length = 500)
init_params = [1.0, 1.0]
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)


## solve the ODE problem
#sol = solve(prob, Midpoint(), saveat=tsteps)
sol = solve(prob, ImplicitMidpoint(), tstops=tsteps)

## print origin data
ode_data = Array(sol)
q_ode_data = ode_data[1,:]
p_ode_data = ode_data[2,:]
#q_ode_data = add_gauss(q_ode_data, 0.01)
#p_ode_data = add_gauss(p_ode_data, 0.01)
dqdt = p_ode_data ./ init_params[1]
dpdt = q_ode_data ./ init_params[2]

plot(tsteps, dqdt)
plot!(tsteps, q_ode_data)
plot(tsteps, dpdt)
plot!(tsteps, p_ode_data)


data = cat(reshape(q_ode_data, 1, :), reshape(p_ode_data, 1, :), dims = 1)
target = cat(reshape(dqdt, 1, :), reshape(dpdt, 1, :), dims = 1)
dataloader = Flux.Data.DataLoader((data, target); batchsize=100, shuffle=true)



## Make a neural network with a NeuralODE layer, where FastChain is a fast neural net structure for NeuralODEs
NN = Chain(Dense(2, 40, tanh), ### Multilayer perceptron for the part we don't know
    Dense(40, 20, tanh),
    Dense(20, 2))

neural_params, re = Flux.destructure(NN)
neural_params
NN_re(x,p) = re(p)(x)

opt = ADAM(0.001)

loss(x, y, p) = mean((NN_re(x, p) .- y) .^ 2)


callback() = println("Loss ODE NET = $(loss(data, target, neural_params))")


epochs = 100
for epoch in 1:epochs
    for (x, y) in dataloader
        optimization_function = ReverseDiff.gradient(p -> loss(x, y, p), neural_params)
        Flux.Optimise.update!(opt, neural_params, optimization_function)
    end
    if epoch % 10 == 1
        callback()
    end
end
callback()
ps = neural_params
NN_re(data[ : , 1],ps)


begin
    function counting(count)
        global count += 1
    end
    count = 0
end


function EulerMethod(u1, i, j)
    u2 = u1 + i .* NN_re(data[ : , j], ps)
    return u2
end

global initial_state = u0

pred = [0,0]
for i in tsteps
    j = counting(count)
    # Euler method
    initial_state = EulerMethod(initial_state, 0.1, j)
    #println(initial_state)
    pred = hcat(pred, initial_state)
end
pred_data = pred[1:2, 2:501]


plot(data[1, :], data[2, :], lw=3, label="Original", xlabel="Position (q)", ylabel="Momentum (p)")
plot!(pred_data[1, :], pred_data[2, :], lw=3, label="Predicted")

H = data[2, :].^2/(2) + data[1, :].^2/(2)
plot(tsteps, H, ylims = (0.8, 1.3))
H_pred = pred_data[2, :].^2/(2) + pred_data[1, :].^2/(2)
plot!(tsteps, H_pred)