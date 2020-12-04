using NCA.FFTW

function ft(xs, ys; mode=+1)
  # @assert ordered_xs
  dx = xs[2] - xs[1]

  x̂s = fftfreq(length(xs), -2pi/dx)
  is = sortperm(x̂s)

  ℯⁱᵠ = dx * exp.(1.0im * xs[1] .* x̂s) * (mode == 1 ? 1 : length(xs))

  return x̂s[is], (ℯⁱᵠ .* (mode == 1 ? fft(ys) : ifft(ys)))[is]
end
