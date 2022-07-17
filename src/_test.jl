






A = [LegendreBasis(2), LegendreBasis(2)]
nn = TensorLayer(A, 1)
f = x -> min(30one(x),x)
u = Float32[2,3,6]
p = Float32[1,2,1,1]
p[1:end]
nn(u,p[1:end])
nn(u,p[1:end])[1]
f(nn(u,p[1:end])[1])