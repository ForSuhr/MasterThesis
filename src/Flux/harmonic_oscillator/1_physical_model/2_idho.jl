# https://diffeqparamestim.sciml.ai/stable/

## using package
using DiffEqFlux, DifferentialEquations, Plots

## define ODEs
function ODEfunc_idho(du,u,params,t) ### du=[̇q,̇p,̇sₑ], u=[q,p,sₑ], params=[m,d,θₒ,c]
  q, p, sₑ = u
  m, d, θₒ, c = params
  ## ODEs
  du[1] = p/m
  du[2] = -q/c-d*p/m
  du[3] = d*(p/m)^2/θₒ
end

"""
m is the mass
c is the spring compliance
d is the damping coefficient
θ_o is the environmental temperature
q is the displacement of the spring
p is the momentum of the mass
s_e is the entropy of the environment
"""

## give initial condition, timespan, parameters, which construct a ODE problem
u0 = [1.0, 1.0, 0.0]
tspan = (0.0, 20.0)
tsteps = range(tspan[1], tspan[2], length = 1000)
init_params = [1.0, 0.4, 1.0, 1.0] ### parameters = [m,d,θₒ,c]
prob = ODEProblem(ODEfunc_idho, u0, tspan, init_params)

## solve the ODE problem once, then add some noise to the solution
sol = solve(prob, Tsit5(), saveat = tsteps)

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
z_axis_ode_data = ode_data[3,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth")
