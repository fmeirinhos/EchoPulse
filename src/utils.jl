# Trapezoidal integration rule
function trapz(x::Vector, y::Vector)
    n = length(x)
    @assert n == length(y) 

    r = zero(typeof(one(eltype(x)) * one(eltype(y))))
    
    if n == 1; return r; end
    
    for i in 2:n
        r += (x[i] - x[i-1]) * (y[i] + y[i-1])
    end
    
    return r/2
end