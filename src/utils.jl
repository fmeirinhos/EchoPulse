"""
Trapezium-rule integration
"""
function integrate(x::AbstractVector, y::AbstractVector)
  if isone(length(x))
    return zero(first(y))
  end
  @inbounds retval = (x[2] - x[1]) * (y[1] + y[2])
  @inbounds @fastmath @simd for i in 2:(length(y) - 1)
    retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
  end
  return 1//2 * retval
end

"""
Real-time collision integral
"""
function integrate(times::Vector,t1,t2,i,j,A::GreenFunction,B::GreenFunction)
  retval = zero(eltype(A))
  @inbounds @fastmath @simd for k in min(i,j):(max(i,j)-1)
      retval += (times[k+1] - times[k]) * (A[t1,k]*B[k,t2] + A[t1,k+1]*B[k+1,t2])
  end
  return 1//2 * sign(j-i) * retval
end
function integrate(times::Vector,t1,t2,i,j,A::Vector{<:GreenFunction},B::GreenFunction)
  retval = zeros(eltype(B), length(A))
  
  for u in eachindex(A)
    @inbounds @fastmath @simd for k in min(i,j):(max(i,j)-1)
      dt = (times[k+1] - times[k])
      retval[u] += dt * (A[u][t1,k]*B[k,t2] + A[u][t1,k+1]*B[k+1,t2])
    end
  end
  return 1//2 * sign(j-i) * retval
end
function integrate(times::Vector,t1,t2,i,j,A::GreenFunction,B::Vector{<:GreenFunction})
  retval = zeros(eltype(A), length(B))
  for u in eachindex(B)
    @inbounds @fastmath @simd for k in min(i,j):(max(i,j)-1)
      dt = (times[k+1] - times[k])
      retval[u] += dt * (A[t1,k]*B[u][k,t2] + A[t1,k+1]*B[u][k+1,t2])
    end
  end
  return 1//2 * sign(j-i) * retval
end

"""
Fourier transformation
"""
function ft(xs, ys; mode=+1)
  # @assert ordered_xs
  dx = xs[2] - xs[1]

  x̂s = fftfreq(length(xs), -2pi/dx)
  is = sortperm(x̂s)

  ℯⁱᵠ = dx * exp.(mode * 1.0im * xs[1] .* x̂s) * (mode == 1 ? 1 : length(xs))

  return x̂s[is], (ℯⁱᵠ .* (mode == 1 ? fft(ys) : ifft(ys)))[is]
end

"""
Builds a GreenFunction from
"""
function G∞(g; n=1)
  L2 = length(g) ÷ 2
  Z_ = zeros(eltype(g), L2*n, L2*n)

  Z_[diagind(Z_, 0)] .= g[L2]

  for i in 1:(L2-1)
    Z_[diagind(Z_, -i)] .= g[L2+i]
    Z_[diagind(Z_, +i)] .= g[L2-i]
  end
  @assert Z_ ≈ -Z_'
  
  return Z_
end