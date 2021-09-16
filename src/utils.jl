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
Fourier transformation
"""
function ft(xs, ys; mode = +1)
    # @assert ordered_xs
    dx = xs[2] - xs[1]

    x̂s = DSP.FFTW.fftfreq(length(xs), -2pi / dx)
    is = sortperm(x̂s)

    ℯⁱᵠ = dx * exp.(mode * 1.0im * xs[1] .* x̂s) * (mode == 1 ? 1 : length(xs))

    return x̂s[is], (ℯⁱᵠ.*(mode == 1 ?  DSP.FFTW.fft(ys) :  DSP.FFTW.ifft(ys)))[is]
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
    x0 =
        nlsolve(
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