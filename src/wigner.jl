import LinearAlgebra: checksquare
import FFTW: fft, fftfreq
import FastTransforms: nufft1
import NFFT: nfft, NFFTPlan
using Plots, LaTeXStrings

"""
    Wigner-Ville transformation

  W_x(ω, T) = i ∫dt x(T + t/2, T - t/2) e^{-i 2π ω t}

The motivation for the Wigner function is that it reduces to the spectral 
density function at all times `T` for stationary processes, yet it is fully 
equivalent to the non-stationary autocorrelation function. Therefore, the 
Wigner function tells us (roughly) how the spectral density changes in time.

## Parameters
  - 2-point correlator (`x`)
  - Time-grid (`ts`): Defaults to a `UnitRange` time-grid
  - Number of ω points (`Nf`): Equidistant number of frequencies.
  Defaults to `size(x,1)`.
  - Transform opt (`transform`): Whether to map `ts` to an equidistant grid.
  Defaults to `true`
  - Fourier opt (`fourier`): Whether to Fourier transform. Defaults to `true`


## References:
https://en.wikipedia.org/wiki/Wigner_distribution_function
http://tftb.nongnu.org
"""
function wigner_transform(x; ts=nothing, Nf=size(x,1), 
  fourier=true, transform=true)

  # Maps the indices of ts to an equidistant time-grid
  function equidistant_map(y)
    equidistant_y = range(y[1], stop=y[end], length=length(y))
    return searchsortedlast.(Ref(equidistant_y), y)
  end

  checksquare(x)

  # Time-grid
  Nt = size(x, 1)
  if !isnothing(ts) && transform
    ts_ = equidistant_map(ts)
  else
    ts_ = 1:Nt
  end
  if !isnothing(ts) @assert length(ts) == Nt end
  
  # Change of basis x(t1, t2) → x′(t1 - t2, (t1 + t2)/2)
  x′ = zero(x)

  for icol=1:Nt
    tᵢ = ts_[icol]

    # For a certain tᵢ ≡ (t1 + t2)/2, τ ≡ (t1 - t2) can be at most τ_max
    τ_max = minimum([tᵢ-1, Nt-tᵢ, Nt÷2-1])
    τs = -τ_max:τ_max
    is = 1 .+ rem.(Nt.+τs, Nt)

    for (i, τᵢ) in zip(is, τs)
      x′[i,icol] = x[tᵢ+τᵢ,tᵢ-τᵢ]
    end
    
    τ = Nt ÷ 2
    if tᵢ <= Nt-τ && tᵢ >= τ+1
      x′[τ+1,icol] = 0.5 * (x[tᵢ+τ,tᵢ-τ] + x[tᵢ-τ,tᵢ+τ])
    end
  end

  if !fourier return x′ end

  # TODO: use just FastTransforms.jl
  if Nf != Nt # Different frequency sampling
    @assert transform "Different frequency sampling requires transformation"

    x′′ = x′
    x′ = zeros(eltype(x′),Nf,Nt)

    mid = isodd(Nt) ? Nt ÷ 2 + 1 : Nt ÷ 2
    p = NFFTPlan(fftfreq(Nf, 0.1), Nt)

    for tᵢ=1:Nt
      x′[:,tᵢ] = nfft(p, [x′′[mid+1:end,tᵢ]; x′′[1:mid,tᵢ]])
    end
  elseif transform # Equidistant FFT
    for tᵢ=1:Nt
      x′[:,tᵢ] = fft(x′[:,tᵢ])
    end
  else # Non-equidistant FFT in time
    @warn "Experimental feature"
    # Map ts to non-equidistant 0:(N-1)
    ts_ = ts .- ts[1]
    ts_ *= (length(ts)-1) / ts_[end]

    for tᵢ=1:Nt
      x′[:,tᵢ] = nufft1(x′[:,tᵢ], ts_, 1e-12)
    end
  end
  return 1.0im * x′
end

function wigner_plot!(ρ, ts; kwargs...)  
  ωs = -2π * fftfreq(size(ρ,1)) / ((ts[end]-ts[1]) / (length(ts)-1)) / 2
  is = sortperm(ωs)

  heatmap(ts, ωs[is], ρ[is,:]; 
    xlabel=L"$T$", 
    ylabel=L"$\omega$",
    kwargs...)
end

# Deprecated
function wigner_transform1(A::Matrix)
  B = zero(A)
  L = size(A, 1)
  
  # Change of basis
  for t1 in 1:L
    for t2 in 1:L
      B[1+((t1-t2)+L)÷2,(t1+t2)÷2] = A[t1,t2]
    end
  end
  
  # Fourier transform
  for i in 1:L
    B[:,i] = fft(B[:,i])
  end
    
  return 1.0im * B
end
