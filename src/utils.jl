"""
Trapezium-rule integration
"""
function integrate(x::AbstractVector, y::AbstractVector)
    if isone(length(y))
        return zero(first(y))
    end
    @inbounds retval = (x[2] - x[1]) * (y[1] + y[2])
    @inbounds @fastmath @simd for i = 2:(length(y)-1)
        retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
    end
    return 1 // 2 * retval
end

"""
Real-time collision integral
"""
function integrate(times::Vector, t1, t2, i, j, A::GreenFunction, B::GreenFunction)
    retval = zero(eltype(A))
    @inbounds @fastmath @simd for k = min(i, j):(max(i, j)-1)
        retval += (times[k+1] - times[k]) * (A[t1, k] * B[k, t2] + A[t1, k+1] * B[k+1, t2])
    end
    return 1 // 2 * sign(j - i) * retval
end

"""
    G∞(g; n=1)

Builds a 2-time GreenFunction constant in the center-of-mass time given g(τ)
"""
function G∞(g; n = 1)
    L2 = (length(g) + 1) ÷ 2
    Z_ = zeros(eltype(g), L2 * n, L2 * n)

    Z_[diagind(Z_, 0)] .= g[L2]

    for i = 1:(L2-1)
        Z_[diagind(Z_, -i)] .= g[L2+i]
        Z_[diagind(Z_, +i)] .= g[L2-i]
    end
    @assert Z_ ≈ -Z_'

    return Z_
end

"""
Finds the fixed point of some function `f` with initial guess `x0`
"""
function fixed_point(f, x0)
    sol = optimize.root(; 
        fun=x -> vec(f(reshape(x, size(x0)...))) - x, 
        x0=vec(x0), 
        method=:anderson,  
        options=Dict(
            :maxiter => 100, 
            :disp => true, 
            :fatol => 1e-8)
    )

    # x0 = reshape(sol["x"], size(x0)...)

    # sol = optimize.root(; 
    #     fun=x -> vec(f(reshape(x, size(x0)...))) - x, 
    #     x0=vec(x0), 
    #     method=:krylov,  
    #     options=Dict(
    #         :maxiter => 20, 
    #         :disp => true, 
    #         :fatol => 1e-8)
    # )

    reshape(sol["x"], size(x0)...), sol["success"]
end

# Gaussian
gaussian(x, x₀, σ) = exp(-0.5 * ((x - x₀) / (σ))^2) / sqrt(2 * π * σ^2)

# Fermi function
nf(ω, β) = 0.5 * (1.0 - tanh(0.5 * β * ω))

# Bose function
nb(ω, β) = !iszero(ω) ? 1.0 / (exp(ω * β) - 1.0) : 0.0
