function stack(dim, eps)
    x = rand(dim)
    for i in 1:eps-1
        x = cat(x, rand(dim), dims = 2)
    end
    return x
end

stack(3, 10)