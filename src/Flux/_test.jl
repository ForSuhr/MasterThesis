


mse_O_NET

mse_structured_O_NET

# Calculate training loss
training_loss_O_NET = sum((ode_data .- trajectory_estimate_O_NET).^2)/200

# Calculate test loss
test_tspan = (20.0, 39.9)
test_tsteps = range(test_tspan[1], test_tspan[2], length = 200)
test_prob = ODEProblem(ODEfunc_udho, u0, test_tspan, init_params)
test_sol = solve(test_prob, ImplicitMidpoint(), tstops = test_tsteps)
## print origin data
test_data = Array(test_sol)
q_test_data = test_data[1,:]
p_test_data = test_data[2,:]
add_gauss(q_test_data, 0.1)
add_gauss(p_test_data, 0.1)
test_loss_O_NET = sum((test_data .- trajectory_estimate_O_NET).^2)/200



