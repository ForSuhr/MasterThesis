using Zygote
θ  = rand(2,2)
function cost(θ_)
  f = x -> sum(θ_*x)
  x = [1.0,2.0]
  sum(ForwardDiff.gradient(f, x))
end
Zygote.gradient(cost,θ) # nothing


using DiffEqFlux, ForwardDiff, LinearAlgebra
nn = FastChain(FastDense(1,32,tanh), FastDense(32,32,tanh), FastDense(32,1))
θ  = initial_params(nn)
function cost(θ_)
  f = x -> nn(x,θ_)[1]
  x = [0f-1]
  sum(ForwardDiff.gradient(f, x))
end
Zygote.gradient(cost,θ) # nothing

