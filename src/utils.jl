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
    # x0 = nlsolve(
    #     x -> f(x) - x,
    #     x0;
    #     show_trace = :true,
    #     method = :anderson,
    #     beta = 0.1,# Since DMFT is kind of unstable we take very conservative steps
    #     m = 1,      # and ignore previous function calls
    #     ftol = 1e-2,
    # ).zero

    nlsolve(
        x -> f(x) - x,
        x0;
        show_trace = :true,
        method = :anderson,
        beta = 0.5,
        ftol = 1e-12,
    ).zero
end

# Gaussian
gaussian(x, x₀, σ) = exp(-0.5 * ((x - x₀) / (σ))^2) / sqrt(2 * π * σ^2)

# Fermi function
nf(ω, β) = 0.5 * (1.0 - tanh(0.5 * β * ω))

# Bose function
nb(ω, β) = !iszero(ω) ? 0.5 * (coth(0.5 * β * ω) - 1.0) : 0.0
