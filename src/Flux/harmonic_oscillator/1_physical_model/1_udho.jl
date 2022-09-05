# https://diffeqparamestim.sciml.ai/stable/

## using package
using DiffEqFlux, DifferentialEquations
using Plots
using Plots.PlotMeasures

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
u0 = [1.0; 1.0]
tspan = (0.0, 20.0)
tsteps = range(tspan[1], tspan[2], length = 1000)
init_params = [1.0, 1.0]
prob = ODEProblem(ODEfunc_udho, u0, tspan, init_params)

## solve the ODE problem
sol = solve(prob, Tsit5(), saveat = tsteps)

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
pyplot()
plot(tsteps, x_axis_ode_data, label="q(t)", xlabel="t", ylabel="x", yguidefontrotation=-90)
plot!(tsteps, y_axis_ode_data, label="p(t)", xlabel="t", ylabel="x", yguidefontrotation=-90)
savefig("udho_time_evolution")
plt = plot(x_axis_ode_data, y_axis_ode_data, yguidefontrotation=-90, xlabel="q", ylabel="p", label="x=(q,p)")
savefig("phase_portrait_canonical_coordinates")
q = ode_data[1,:]
p = ode_data[2,:]
m, c = init_params
H = p.^2/(2m) + q.^2/(2c)
plt = plot(tsteps, round.(H, digits=2), ylims = (0,1.5), label="H(t)", xlabel="t", ylabel="H", yguidefontrotation=-90)
savefig("udho_time_evolution_Hamiltonian")
