# https://diffeqparamestim.sciml.ai/stable/tutorials/ODE_inference/

## using package
using Revise
using DiffEqFlux, DifferentialEquations, LinearAlgebra, Plots
using DiffEqParamEstim
using RecursiveArrayTools # for VectorOfArray
using LeastSquaresOptim # for LeastSquaresProblem, https://github.com/matthieugomez/LeastSquaresOptim.jl

## define ODEs
function ODEfunc_ndho(du,u,params,t) ### du=[̇q,̇p,̇sd,̇sₑ], u=[q,p,sd,sₑ], p=[m,d,c,θₒ,θd,α]
  q, p, sd, sₑ = u
  m, d, c, θₒ, θd, α = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c-d*p/m
  du[3] = d*((p/m)^2)/θd-α*(θd-θₒ)/θd
  du[4] = α*(θd-θₒ)/θₒ
end

"""
m is the mass
c is the spring compliance
d is the damping coefficient
θ_o is the environmental temperature
θ_d is the temperature of the damper 
q is the displacement of the spring
p is the momentum of the mass
s_e is the entropy of the environment
α is the heat transfer coefficient
"""

## give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0, 1.0, 0.0, 0.0]
tspan = (0.0, 20.0)
tsteps = range(tspan[1], tspan[2], length = 1000)
init_params = [1.0; 0.4; 1.0; 2.0; 3.0; 5.0] ### parameters = [m,d,c,θₒ,θd,α]
prob = ODEProblem(ODEfunc_ndho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, Tsit5(), saveat = tsteps)
t = collect(range(0, stop=20, length=1000))
randomized = VectorOfArray([(sol(t[i]) + .02randn(4)) for i in 1:length(t)]) ## sol(t[i]) means u(t) at the timepoint t=i
data = convert(Array,randomized) ## convert the randomized data from a vector into an array 
plot(sol)
scatter!(t,data')

## build a LeastSquaresOptim object so that we can solve it by using LeastSquaresOptim.jl in the following
loss_function = build_lsoptim_objective(prob,t,data,Tsit5())
## guess an initial set of parameters
guess_params = [1.0; 0.4; 1.0; 2.0; 3.0; 14.0]
## using LeastSquaresOptim.optimize!
res = optimize!(LeastSquaresProblem(x = guess_params, f! = loss_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR()), iterations = 1000)

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
z1_axis_ode_data = ode_data[3,:]
z2_axis_ode_data = ode_data[4,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, z1_axis_ode_data, label="Original")
plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_ode_data, label="Original")

## use the estimated parameters to restore the system, then compare it to the origin
estimated_prob_ODEfunc_ndho = ODEProblem(ODEfunc_ndho, u0, tspan, est_params)
sol = solve(estimated_prob_ODEfunc_ndho, Tsit5(), saveat = tsteps)
est_data = Array(sol)
x_axis_est_data = est_data[1,:]
y_axis_est_data = est_data[2,:]
z1_axis_est_data = est_data[3,:]
z2_axis_est_data = est_data[4,:]
plt = plot!(x_axis_est_data, y_axis_est_data, z1_axis_est_data, label="Estimated")
plt = plot!(x_axis_est_data, y_axis_est_data, z2_axis_est_data, label="Estimated")
