
using DiffEqProblemLibrary.ODEProblemLibrary, Plots
ODEProblemLibrary.importodeproblems()
prob = ODEProblemLibrary.prob_ode_linear
sol = solve(prob)
plot(sol)