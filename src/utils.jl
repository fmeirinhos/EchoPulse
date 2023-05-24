# Fermi-Dirac distribution function
nf(ω, β) = 0.5 * (1.0 - tanh(0.5 * β * ω))

# Bose-Einstein distribution function
nb(ω, β) = !iszero(ω) ? 0.5 * (coth(0.5 * β * ω) - 1.0) : 0.0

# Bethe lattice density of states (∫dε bethe_dos(ε) = 1)
bethe_dos(ε) = abs(ε) <= 2 ? 1 / pi * sqrt(1 - (ε / 2)^2) : 0.0

# Free photon gas density of states
photon_dos(ε; cutoff) = (ε < 0 ? 0.0 : ε < cutoff ? ε : ε * exp(-(ε - cutoff) / cutoff))

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
Inverts *fermionic* Green functions in the Retarded-Advanced-Keldysh representation
"""
@inline function inv_RKA(A)
    newtype = eltype(A)
    idet = inv(A[1] * A[4])
    @inbounds return newtype[A[4]*idet -(A[3] * idet); zero(newtype) A[1]*idet]
end

"""
    wigner_transform_itp(x::AbstractMatrix, ts::Vector; fourier=true)

Interpolates `x` on an equidistant mesh with the same boundaries and length as `ts` and calls [`wigner_transform`](@ref)
"""
function wigner_transform_itp(x::AbstractMatrix, ts::Vector; fourier = true, ts_lin = range(first(ts), last(ts); length = length(ts)))
    itp = interpolate((ts, ts), x, Gridded(Linear()))
    return wigner_transform([itp(t1, t2) for t1 in ts_lin, t2 in ts_lin]; ts = ts_lin, fourier = fourier)
end

# Calculates the Kondo temperature
kondo_temperature(V0, ε0, ρ0, D0, N) = (W0 = V0^2 * ρ0; D0 * (N * W0 / D0)^(1 / N) * exp(ε0 / (N * W0)))

kondo_temperature2(V0, ε0, ρ0, D0, N, U0=Inf) = (J = abs(-2 * V0^2 / ε0 * (isinf(U0) ? 1.0 : U0 / (U0 + ε0))); D0 * (N * J * ρ0)^(1 / N) * exp(-1 / (N * J * ρ0)))

function kondo_temperature_from_fit(ωs, GdRω, ε0f, U0)
    if isinf(U0)
        fit = curve_fit((ω, p) -> (@. p[1] * p[2] / ((ω - p[3])^2 + p[2]^2) + p[4] * p[5] / ((ω - p[6])^2 + p[5]^2)), ωs, -imag(GdRω), [1.0, 0.2, ε0f, 1.0, 0.03, 0.0])
        fit.converged ? (abs(fit.param[3]) < abs(fit.param[6]) ? abs(fit.param[2]) : abs(fit.param[5])) : 0.015
    else
        fit = curve_fit((ω, p) -> (@. p[1] * p[2] / ((ω - p[3])^2 + p[2]^2) + p[4] * p[5] / ((ω - p[6])^2 + p[5]^2) + p[7] * p[8] / ((ω - p[9])^2 + p[8]^2)), ωs, -imag(GdRω), [1.0, 0.2, ε0f, 1.0, 0.03, 0.0, 1.0, 0.2, ε0f + U0])
        if fit.converged
            if abs(fit.param[3]) < abs(fit.param[6])
                if abs(fit.param[3]) < abs(fit.param[9])
                    return abs(fit.param[2])
                else
                    return abs(fit.param[8])
                end
            else
                if abs(fit.param[6]) < abs(fit.param[9])
                    return abs(fit.param[5])
                else
                    return abs(fit.param[8])
                end
            end
        else
            @error "Kondo temperature not found"
        end
    end
end
