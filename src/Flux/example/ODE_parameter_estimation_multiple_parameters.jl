# https://diffeqparamestim.sciml.ai/stable/tutorials/ODE_inference/

## using package
using DiffEqFlux, DifferentialEquations, Plots
using DiffEqParamEstim
using RecursiveArrayTools # for VectorOfArray
using LeastSquaresOptim # for LeastSquaresProblem

## define ODEs
function f2(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

## give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f2,u0,tspan,p)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob,Tsit5())
t = collect(range(0,stop=10,length=200))
randomized = VectorOfArray([(sol(t[i]) + .02randn(2)) for i in 1:length(t)]) ## sol(t[i]) means u(t) at the timepoint t=i
data = convert(Array,randomized) ## convert the randomized data from a vector into an array 
plot(sol)
scatter!(t,data')

"""
## build a loss function
cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS()) ## BFGS is a quasi Newton method


## the kwarg differ_weight decides the contribution of the differencing loss to the total loss
cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data,differ_weight=0.3,data_weight=0.7),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
"""

## build a LeastSquaresOptim object so that we can solve it by using LeastSquaresOptim.jl in the following
cost_function = build_lsoptim_objective(prob,t,data,Tsit5())
## using LeastSquaresOptim.optimize!
### https://github.com/matthieugomez/LeastSquaresOptim.jl
"""
x is an initial set of parameters.
f!(out, x) that writes f(x) in out.
the option output_length to specify the length of the output vector.
Optionally, g! a function such that g!(out, x) writes the jacobian at x in out. Otherwise, the jacobian will be computed following the :autodiff argument.
"""
x = [1.5,0.8,2.8,1.2]
res = optimize!(LeastSquaresProblem(x = x, f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR()))

println(res.minimizer)

