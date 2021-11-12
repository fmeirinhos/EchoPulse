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
Trapezium rule integration with generator y (important when `y` can't be put into a vector)
"""
function integrate(x::AbstractVector, y)
    if isone(length(y))
        return zero(first(y))
    end
    yold = iterate(y, 0)[1]
    ynew = iterate(y, 1)[1]
    @inbounds retval = (x[2] - x[1]) * (yold + ynew)
    @inbounds @fastmath @simd for i = 2:(length(y)-1)
        yold = ynew
        ynew = iterate(y, i)[1]
        retval += (x[i+1] - x[i]) * (yold + ynew)
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
Real-time collision integral with factorised plane-wave
"""
function integrate(times::Vector, t1, t2, i, j, A::GreenFunction, B::GreenFunction, ε)
    retval = zero(eltype(A))
    @inbounds @fastmath @simd for k = min(i, j):(max(i, j)-1)
        e0 = exp(+im * ε * (times[t1] - times[k]))
        e1 = exp(+im * ε * (times[t1] - times[k+1]))

        retval += (times[k+1] - times[k]) * (A[t1, k] * e0 * B[k, t2] + A[t1, k+1] * e1 * B[k+1, t2])
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
    x0 = nlsolve(
        x -> f(x) - x,
        x0;
        show_trace = :true,
        method = :anderson,
        beta = 0.1,# Since DMFT is kind of unstable we take very conservative steps
        m = 1,      # and ignore previous function calls
        ftol = 1e-2,
    ).zero

    nlsolve(
        x -> f(x) - x,
        x0;
        show_trace = :true,
        method = :anderson,
        beta = 0.5,
        ftol = 1e-6,
    ).zero
end

# Gaussian
gaussian(x, x₀, σ) = exp(-0.5 * ((x - x₀) / (σ))^2) / sqrt(2 * π * σ^2)

# Fermi function
nf(ω, β) = 0.5 * (1.0 - tanh(0.5 * β * ω))

# Bose function
nb(ω, β) = !iszero(ω) ? 0.5 * (coth(0.5 * β * ω) - 1.0) : 0.0
