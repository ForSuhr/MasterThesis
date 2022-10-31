using Flux
using Zygote
using ReverseDiff
using Statistics

W = rand(2, 3)
b = rand(2)

predict(x) = sigmoid((W * x) .+ b)

loss(x, y) = mean(abs2, (predict(x) .- y))
loss(rand(3), rand(2))

function stack(dim, eps)
    x = rand(dim)
    for i in 1:eps-1
        x = cat(x, rand(dim), dims = 2)
    end
    return x
end
x = stack(3, 10)
y = stack(2, 10)

θ = Flux.params(W, b)
gs(x, y) = Zygote.gradient(() -> loss(x, y), θ)


dataloader = Flux.Data.DataLoader((x, y), batchsize = 10)

# Learning Rate
α = 0.1 
# opt = Flux.Descent(0.1)
opt = Flux.Optimise.ADAM(0.01, (0.9, 0.99))
for (x, y) in dataloader
    for θ in (W, b) 
        # θ .-= α * gs(x, y)[θ]
        Flux.Optimise.update!(opt, θ, gs(x, y)[θ])
    end
end

θ
W
b
l = loss(x, y)