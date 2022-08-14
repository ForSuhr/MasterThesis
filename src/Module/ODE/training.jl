using Optimization
using OptimizationFlux


function train(loss_function, params, callback; learning_rate=0.01, maxiters=100, adtype = Optimization.AutoZygote())
  optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, params)
  @time res = Optimization.solve(optprob, ADAM(learning_rate), callback = callback, maxiters = maxiters)
  return res
end