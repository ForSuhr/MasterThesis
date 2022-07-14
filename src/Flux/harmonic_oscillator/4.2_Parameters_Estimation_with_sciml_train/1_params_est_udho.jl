# isothermal damped harmonic oscillator ODE
# https://diffeqflux.sciml.ai/v1.47/examples/stiff_ode_fit/

## using packages
using Revise
using DifferentialEquations, DiffEqFlux, LinearAlgebra
using ForwardDiff
using DiffEqBase: UJacobianWrapper
using Plots

## define a structured neural ODE
function ODEfunc_uho(du,u,params,t) ### du=[̇q,̇p], u=[q,p], p=[m,c]
  ## conversion
  q, p = u
  m, c = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c
  nothing
end

"""
m is the mass
c is the spring compliance
d is the damping coefficient
q is the displacement of the spring
p is the momentum of the mass
"""

## set initial condition
u0 = Float64[1.0; 1.0] ### u0 = [q,p]
initial_parameters = Float64[1.0; 1.0] ### parameters = [m,c]
datasize = 1000
tspan = (0.0f0, 20.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

## solve the ODEProblem
prob_ODEfunc_uho_initial = ODEProblem(ODEfunc_uho, u0, tspan, initial_parameters)
sol_initial = solve(prob_ODEfunc_uho_initial, Tsit5(), saveat = tsteps)
ts = sol_initial.t
us = sol_initial.u
Js = map(u->I + 0.1*ForwardDiff.jacobian(UJacobianWrapper(ODEfunc_uho, 0.0, initial_parameters), u), us)

## define a predict adjoint
function predict_adjoint(p)
  p = exp.(p)
  _prob = remake(prob_ODEfunc_uho_initial,p=p) ## remake means use new parameters to replace the old parameters
  Array(solve(_prob,Rosenbrock23(autodiff=false),saveat=ts,sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

## define a loss adjoint
function loss_adjoint(p)
  prediction = predict_adjoint(p)
  prediction = [prediction[:, i] for i in axes(prediction, 2)]
  diff = map((J,u,data) -> J * (abs2.(u .- data)) , Js, prediction, us)
  loss = sum(abs, sum(diff)) |> sqrt
  loss, prediction
end

## define a callback function
cb = function (p,l,pred) #callback function to observe training
  println("Loss: $l")
  println("Parameters: $(exp.(p))")
  # using `remake` to re-create our `prob` with current parameters `p`
  plot(solve(remake(prob_ODEfunc_uho_initial, p=exp.(p)), Rosenbrock23())) |> display
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

## guess a set of parameters
guess_initp = Float64[0.5; 0.5]
# Display the ODE with the initial parameter values.
cb(guess_initp,loss_adjoint(guess_initp)...)
## use a combination of ADAM and BFGS to minimize the loss function
res1 = DiffEqFlux.sciml_train(loss_adjoint, guess_initp, ADAM(0.01), cb = cb, maxiters = 300)
res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.u, BFGS(), cb = cb, maxiters = 30, allow_f_increases=true)
println("Ground truth: $(initial_parameters)\nFinal parameters: $(round.(exp.(res2.u), sigdigits=5))\nError: $(round(norm(exp.(res2.u) - initial_parameters) ./ norm(initial_parameters) .* 100, sigdigits=3))%")

## get the estimated parameters
estimated_params = (round.(exp.(res2.u), sigdigits=5))

## print origin data
ode_data = Array(sol_initial)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Original")

## use the estimated parameters to restore the system, then compare it to the origin
prob_ODEfunc_uho = ODEProblem(ODEfunc_uho, u0, tspan, estimated_params)
sol = solve(prob_ODEfunc_uho, Tsit5(), saveat = tsteps)
est_data = Array(sol)
x_axis_est_data = est_data[1,:]
y_axis_est_data = est_data[2,:]
plt = plot!(x_axis_est_data, y_axis_est_data, label="Estimated")

