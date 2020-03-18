using FFTW, NFFT
using Plots

# Wigner transform ρ(X, ω) = i ∫dt e^{i \omega t} ρ(X + t/2, X - t/2)
function wigner_transform(x; Nf=nothing)
  Nt = size(x, 1)
  tfr = zeros(ComplexF64, Nt, Nt)
  
  # Change of basis
  for tᵢ = 1:Nt           
    τ_max = minimum([tᵢ - 1, Nt - tᵢ, round(Int, Nt / 2) - 1])
    τ = -τ_max:τ_max
    is = 1 .+ rem.(Nt .+ τ, Nt)
    
    for (i, τᵢ) in zip(is, τ)
      tfr[i,tᵢ] = x[tᵢ+τᵢ,tᵢ-τᵢ]
    end
    
    τ = round(Int, Nt / 2); 
    if tᵢ <= Nt - τ && tᵢ >= τ + 1
      tfr[τ+1,tᵢ] = 0.5 * (x[tᵢ+τ,tᵢ-τ] + x[tᵢ-τ,tᵢ+τ])
    end
  end
  
  # Fourier Transform
  if isnothing(Nf)
    for tᵢ=1:Nt
      tfr[:,tᵢ]=fft(tfr[:,tᵢ])
    end
    return 1.0im * tfr
  else
    tfr′ = zeros(ComplexF64,Nf,Nt)

    p = NFFTPlan(fftfreq(Nf), Nt)

    for tᵢ=1:Nt      
      mid = isodd(Nt) ? Nt ÷ 2 + 1 : Nt ÷ 2
      tfr′[:,tᵢ]= nfft(p, [tfr[mid+1:end,tᵢ]; tfr[1:mid,tᵢ]])
    end
    
    return 1.0im * tfr′
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
  ωs = fftfreq(size(ρ,1)) / (ts[2] - ts[1]) * 2π
  is = sortperm(ωs)
  heatmap(ts, ωs[is], real(ρ)[is,:]; 
    xlabel=L"$T$", 
    ylabel=L"$\omega$",
    kwargs...)
end
