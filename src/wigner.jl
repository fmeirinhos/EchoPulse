import LinearAlgebra: checksquare
import FFTW: fft, fftfreq
# import FastTransforms: nufft1, nufft2
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
  - Fourier opt (`fourier`): Whether to Fourier transform. Defaults to `true`

## References:
https://en.wikipedia.org/wiki/Wigner_distribution_function
http://tftb.nongnu.org
"""
function wigner_transform(x; ts=1:size(x,1), Nf=size(x,1), fourier=true)

  checksquare(x)

  Nt = size(x, 1)
  
  # Bin time-grid `ts`
  @assert length(ts) == Nt
  ts_ = equidistant_map(ts)

  # Change of basis x(t1, t2) → x′(t1 - t2, (t1 + t2)/2)
  x′ = zero(x)

  for icol=1:Nt
    tᵢ = ts_[icol] # Binning of ts

    # For a certain tᵢ ≡ (t1 + t2)/2, τ ≡ (t1 - t2) can be at most τ_max
    τ_max = minimum([tᵢ-1, Nt-tᵢ, Nt÷2-1])

    # NOTE: Is the binning done correctly??
    # τ_max = minimum([icol-1, Nt-icol, Nt÷2-1])
    # is_ = [icol, Nt-icol+1, Nt÷2]
    # τ_max = is_[argmin(ts[is_])] - 1
    # x′[i,icol] = x[ts_[icol+τᵢ],ts_[icol-τᵢ]]

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

  if Nf != Nt # Different frequency sampling
    # for tᵢ=1:Nt
    #   x′[:,tᵢ] = nufft2(x′[:,tᵢ], fftfreq(Nt, Nt/Nf), 1e-12)
    # end

    x′′ = x′
    x′ = zeros(eltype(x′),Nf,Nt)

    mid = isodd(Nt) ? Nt÷2 + 1 : Nt÷2
    p = NFFTPlan(fftfreq(Nf, 1.0), Nt)

    for tᵢ=1:Nt
      x′[:,tᵢ] = nfft(p, [x′′[mid+1:end,tᵢ]; x′′[1:mid,tᵢ]])
    end
  else # Equidistant FFT
    for tᵢ=1:Nt
      x′[:,tᵢ] = fft(x′[:,tᵢ])
    end
  end
  # else # Non-equidistant FFT in time
  #   @warn "Experimental feature. Requires change of basis without binining"
  #   # Map ts to non-equidistant 0:(N-1)
  #   ts_ = ts .- ts[1]
  #   ts_ *= (length(ts)-1) / ts_[end]

  #   for tᵢ=1:Nt
  #     x′[:,tᵢ] = nufft1(x′[:,tᵢ], ts_, 1e-12)
  #   end   # Frequencies are equispaced
  # end
  return 1.0im * x′
end

function wigner_plot!(ρ, ts; transform=true, kwargs...)
  if transform
    ts = range(ts[1], stop=ts[end], length=length(ts))
  end
  ωs = -2π * fftfreq(size(ρ,1)) / ((ts[end]-ts[1]) / (length(ts)-1)) / 2
  is = sortperm(ωs)

  heatmap(ts, ωs[is], ρ[is,:]; 
    xlabel=L"$T$", 
    ylabel=L"$\omega$",
    kwargs...)
end

# Maps the indices of ts to an equidistant time-grid
function equidistant_map(x)
  equidistant_y = range(x[1], stop=x[end], length=length(x))
  return searchsortedlast.(Ref(equidistant_y), x)
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
