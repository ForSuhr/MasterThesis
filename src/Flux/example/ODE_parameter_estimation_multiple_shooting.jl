# https://diffeqparamestim.sciml.ai/stable/tutorials/ODE_inference/

##
using DiffEqFlux, DifferentialEquations, Plots
using DiffEqParamEstim
using RecursiveArrayTools # for VectorOfArray

##
function ms_f(du,u,p,t)
  dx = p[1]*u[1] - p[2]*u[1]*u[2]
  dy = -3*u[2] + u[1]*u[2]
end
ms_u0 = [1.0;1.0]
tspan = (0.0,10.0)
ms_p = [1.5,1.0]
ms_prob = ODEProblem(ms_f,ms_u0,tspan,ms_p)
t = collect(range(0,stop=10,length=200))
data = Array(solve(ms_prob,Tsit5(),saveat=t,abstol=1e-12,reltol=1e-12))
bound = Tuple{Float64, Float64}[(0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),(0, 10),(0, 10)]


ms_obj = multiple_shooting_objective(ms_prob,Tsit5(),L2Loss(t,data);discontinuity_weight=1.0,abstol=1e-12,reltol=1e-12)

##
using BlackBoxOptim

result = bboptimize(ms_obj;SearchRange = bound, MaxSteps = 21e3)
result.archive_output.best_candidate[end-1:end]

##
two_stage_obj = two_stage_method(ms_prob,t,data)
result = Optim.optimize(two_stage_obj, [1.3,0.8,2.8,1.2], Optim.BFGS())
