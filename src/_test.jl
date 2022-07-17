ode_data
aaa = ode_data[1,:]




x = [:a,:b,:c,:d,:e,:f]
n = 2
reshape(x, (n, div(length(x), n)))

aaa
n = 2
ccc = reshape(aaa, (n, div(length(aaa), n)))
print(ccc)
print(ccc[:,1])


tsteps[1]


dddd= [1,2,1,3,4,5,4,3,2,1]
unique(dddd)