# https://diffeqparamestim.sciml.ai/stable/

## using package
using DiffEqFlux, DifferentialEquations, LinearAlgebra, Plots

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

## print origin data
ode_data = Array(sol)
x_axis_ode_data = ode_data[1,:]
y_axis_ode_data = ode_data[2,:]
z1_axis_ode_data = ode_data[3,:]
z2_axis_ode_data = ode_data[4,:]
plt = plot(x_axis_ode_data, y_axis_ode_data, z1_axis_ode_data, label="Ground truth")
plt = plot!(x_axis_ode_data, y_axis_ode_data, z2_axis_ode_data, label="Ground truth")
