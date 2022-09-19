"""
    G∞(g)

Builds a 2-time GreenFunction constant in the center-of-mass time given g≷(τ)
"""
function G∞(g::Vector)
    L2 = (length(g) + 1) ÷ 2
    Z_ = zeros(eltype(g), L2, L2)

    Z_[diagind(Z_, 0)] .= 0.5(g[L2] - g[L2]')

    for i = 1:(L2-1)
        Z_[diagind(Z_, -i)] .= -g[L2-i]' #g[L2+i]
        Z_[diagind(Z_, +i)] .= +g[L2-i]
    end
    @assert Z_ ≈ -Z_'

    return Z_
end

"""
Finds the fixed point of some function `f` with initial guess `x0`
"""
function fixed_point(f, x0)
    nlsolve(
        x -> f(x) - x,
        x0;
        show_trace = :true,
        method = :anderson,
        iterations = 300,
        beta = 0.6, 
        m=3,
        ftol=1e-13
    ).zero
end

# Fermi-Dirac distribution function
nf(ω, β) = 0.5 * (1.0 - tanh(0.5 * β * ω))

# Bose-Einstein distribution function
nb(ω, β) = !iszero(ω) ? 0.5 * (coth(0.5 * β * ω) - 1.0) : 0.0

# Bethe lattice density of states (∫dε bethe_dos(ε) = 1)
bethe_dos(ε) = abs(ε) <= 2 ? 1 / pi * sqrt(1 - (ε / 2)^2) : 0.0

"""
    wigner_transform_itp(x::AbstractMatrix, ts::Vector; fourier=true)

Interpolates `x` on an equidistant mesh with the same boundaries and length as `ts` and calls [`wigner_transform`](@ref)
"""
function wigner_transform_itp(x::AbstractMatrix, ts::Vector; fourier = true)
    ts_lin = range(first(ts), last(ts); length = length(ts))
    itp = interpolate((ts, ts), x, Gridded(Linear()))
    return wigner_transform([itp(t1, t2) for t1 in ts_lin, t2 in ts_lin]; ts = ts_lin, fourier = fourier)
end

# Inverts *fermionic* Green functions in the Retarded-Advanced-Keldysh representation
@inline function inv_RKA(A)
    newtype = eltype(A)
    idet = inv(A[1] * A[4])
    @inbounds return newtype[A[4]*idet -(A[3] * idet); zero(newtype) A[1]*idet]
end
