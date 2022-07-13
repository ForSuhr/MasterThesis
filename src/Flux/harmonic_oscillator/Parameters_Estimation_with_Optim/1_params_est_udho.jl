# https://diffeqparamestim.sciml.ai/stable/tutorials/ODE_inference/

## using package
using DiffEqFlux, DifferentialEquations, Plots
using DiffEqParamEstim
using RecursiveArrayTools # for VectorOfArray
using LeastSquaresOptim # for LeastSquaresProblem, https://github.com/matthieugomez/LeastSquaresOptim.jl

## define ODEs
function ODEfunc_uho(du,u,params,t)
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
tsteps = range(tspan[1], tspan[2], length = 1000)
init_params = [1.5, 1.0]
prob = ODEProblem(ODEfunc_uho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, Tsit5(), saveat = tsteps)
t = collect(range(0, stop=20, length=1000))
randomized = VectorOfArray([(sol(t[i]) + .02randn(2)) for i in 1:length(t)]) ## sol(t[i]) means u(t) at the timepoint t=i
data = convert(Array,randomized) ## convert the randomized data from a vector into an array 
plot(sol)
scatter!(t,data')

## build a LeastSquaresOptim object so that we can solve it by using LeastSquaresOptim.jl in the following
loss_function = build_lsoptim_objective(prob, t, data, Tsit5())
## guess an initial set of parameters
guess_params = [1.5,0.8]
## using LeastSquaresOptim.optimize!
res = optimize!(LeastSquaresProblem(x = guess_params, f! = loss_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR()))


est_params = res.minimizer
println("Ground truth: $(init_params)\nEstimated parameters: $(round.(est_params, sigdigits=5))\nError: $(round(norm(est_params - init_params) ./ norm(init_params) .* 100, sigdigits=3))%")


"""
x is an initial set of parameters.
f!(out, x) that writes f(x) in out.
the option output_length to specify the length of the output vector.
Optionally, g! a function such that g!(out, x) writes the jacobian at x in out. Otherwise, the jacobian will be computed following the :autodiff argument.
"""

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Original")

## use the estimated parameters to restore the system, then compare it to the origin
prob_ODEfunc_uho = ODEProblem(ODEfunc_uho, u0, tspan, est_params)
sol = solve(prob_ODEfunc_uho, Tsit5(), saveat = tsteps)
est_data = Array(sol)
x_axis_est_data = est_data[1,:]
y_axis_est_data = est_data[2,:]
plt = plot!(x_axis_est_data, y_axis_est_data, label="Estimated")
