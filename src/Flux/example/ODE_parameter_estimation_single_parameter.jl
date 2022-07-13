# https://diffeqparamestim.sciml.ai/stable/tutorials/ODE_inference/

##
using DiffEqFlux, DifferentialEquations, Plots
using DiffEqParamEstim
using RecursiveArrayTools # for VectorOfArray

## creat a function that generate ground truth data
function f(du,u,p,t)
    du[1] = dx = p[1]*u[1] - u[1]*u[2]
    du[2] = dy = -3*u[2] + u[1]*u[2]
  end

## give a initial condition, timestep and parameter, which construct a ODE problem
u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob,Tsit5())
t = collect(range(0,stop=10,length=200))
randomized = VectorOfArray([(sol(t[i]) + .02randn(2)) for i in 1:length(t)]) ## sol(t[i]) means u(t) at the timepoint t=i
data = convert(Array,randomized) ## convert the randomized data from a vector into an array 
plot(sol)
scatter!(t,data')

## build a cost function aka loss function
cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)

## visualize the loss function
vals = 0.0:0.1:10.0
plotly()
plot(vals,[cost_function(i) for i in vals],yscale=:log10,
     xaxis = "Parameter", yaxis = "Cost", title = "1-Parameter Cost Function",
     lw = 3)

## use optimize() to get the parameter
using Optim
result = optimize(cost_function, 0.0, 10.0) ## 0.0 and 10.0 means you have selected the interval (0.0,10.0) to estimate the parameters
result = optimize(cost_function, [1.42], BFGS())

##
lower = [0.0]
upper = [3.0]
result = optimize(cost_function, lower, upper, [1.42], Fminbox(BFGS()))

##
function f2(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f2,u0,tspan,p)

cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())

cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data,differ_weight=0.3,data_weight=0.7),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())

cost_function = build_lsoptim_objective(prob,t,data,Tsit5())

using LeastSquaresOptim # for LeastSquaresProblem
x = [1.3,0.8,2.8,1.2]
res = optimize!(LeastSquaresProblem(x = x, f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR()))

println(res.minimizer)

