module IOHelper
using JLD

function SaveParams(path, name, θ)
    save(path, name, θ)
end

function LoadParams(path, name)
    return load(path, name)
end

export SaveParams, LoadParams

end
