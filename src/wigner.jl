import LinearAlgebra: checksquare
import FFTW: fft, fftfreq
import NFFT: nfft, NFFTPlan
import DSP: tukey
using Plots, LaTeXStrings

# Wigner transform ρ(ω, T) = i ∫dt e^{-i ω t} ρ(T + t/2, T - t/2)
function wigner_transform(x; fourier=true, Nf=nothing, α=nothing)
  checksquare(x)

  Nt = size(x, 1)
  x′ = zero(x)
  
  # Change of basis x(t1, t2) → x′((t1 + t2)/2, t1 - t2)
  for tᵢ = 1:Nt
    # For a certain tᵢ ≡ (t1 + t2)/2, τ ≡ (t1 - t2) can be at most τ_max
    τ_max = minimum([tᵢ - 1, Nt - tᵢ, Nt ÷ 2 - 1])
    τs = -τ_max:τ_max
    is = 1 .+ rem.(Nt .+ τs, Nt)
    
    for (i, τᵢ) in zip(is, τs)
      x′[i,tᵢ] = x[tᵢ+τᵢ,tᵢ-τᵢ]
    end
    
    τ = Nt ÷ 2 
    if tᵢ <= Nt - τ && tᵢ >= τ + 1
      x′[τ+1,tᵢ] = 0.5 * (x[tᵢ+τ,tᵢ-τ] + x[tᵢ-τ,tᵢ+τ])
    end
  end

  # Wrap signal
  if !isnothing(α)
    w = tukey(Nt, α)
    x′ = mapslices(c-> c.* w, x′, dims=[1])
  end

  if !fourier
    return trf
  end

  # Fourier Transform
  if isnothing(Nf)
    for tᵢ=1:Nt
      x′[:,tᵢ] = fft(x′[:,tᵢ])
    end
    return 1.0im * x′
  else
    x′′ = zeros(ComplexF64,Nf,Nt)
    mid = isodd(Nt) ? Nt ÷ 2 + 1 : Nt ÷ 2

    p = NFFTPlan(fftfreq(Nf), Nt)

    for tᵢ=1:Nt
      x′′[:,tᵢ] = nfft(p, [x′[mid+1:end,tᵢ]; x′[1:mid,tᵢ]])
    end
    
    return 1.0im * x′′
  end
end

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

function wigner_plot!(ρ, ts; kwargs...)  
  ωs = -2π * fftfreq(size(ρ,1)) / (ts[2] - ts[1]) / 2
  is = sortperm(ωs)

  heatmap(ts, ωs[is], ρ[is,:]; 
    xlabel=L"$T$", 
    ylabel=L"$\omega$",
    kwargs...)
end
