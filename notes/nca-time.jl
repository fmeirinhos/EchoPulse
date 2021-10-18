### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 9206bdaa-2cf6-11ec-17c9-1b978be94cdb
begin
	using Pkg; Pkg.activate() # Just some Pluto bs
	
	using Plots
	using FFTW
	using PyCall
	optimize = pyimport("scipy.optimize")
	
	# Fourier transform
	function ft(xs, ys; mode = +1)
		@assert issorted(xs)

		L = length(xs)
		dx = xs[2] - xs[1]
		
		# FFT
		ŷs = (mode == -1 ? fft(ys) : ifft(ys))
		
		# Because the FFT calculates the transform as
		#   ỹ_k = \sum_j e^{±2pi i j k/n} y_j, from j=0 to j=n-1,
		# we need to transform this into the time and frequency units, 
		# which ends up scaling the frequencies `x̂s` by (2pi / dx).
		x̂s = fftfreq(L, 2π / dx)

		# The resulting Fourier transform also picks up a phase
		ℯⁱᵠ = (mode == -1 ? 1 : L) * dx * exp.(mode * 1.0im * xs[1] .* x̂s) 

		return circshift(x̂s, L ÷ 2), circshift(ℯⁱᵠ .* ŷs, L ÷ 2)
	end
	
	# Nonlinear solve
	function nlsolve(f, x0; method, iterations=20, show_trace=true, kwargs...)
		sol = optimize.root(; 
			fun=x -> vec(f(reshape(x, size(x0)...))), 
			x0=vec(x0), 
			method=method,  
			options=Dict(:maxiter => iterations, :disp => show_trace, kwargs...)
		)

		return reshape(sol["x"], size(x0)...)
	end
end;

# ╔═╡ d271c87b-f68b-4ae5-a1d3-576207ed7bbb
begin
	# Gaussian
	gaussian(x, x₀, σ) = exp(-0.5 * ((x - x₀) / (σ))^2) / sqrt(2 * π * σ^2)
	
	# Fermi function
	nf(ω, β) = 0.5 * (1.0 - tanh(0.5 * β * ω))
	
	# Bose function
	nb(ω, β) = !iszero(ω) ? 1.0 / (exp(ω * β) - 1.0) : 0.0
end;

# ╔═╡ 01e9c802-cbb0-409f-ba23-ef8510c59b1c
nca = let
	β = 1.0e3
	model = (V₀=sqrt(0.1), ε₀f=-0.4, Γ₀=0sqrt(0.01))
	mesh = range(-6.0, 6.0, length=2^14 - 1)
	
	ρc = 2π * gaussian.(mesh, 0.0, 1.0)
	ργ = 2π * gaussian.(mesh, 5e-2, 2e-2) .* mesh
	ργ ./= sum(ργ) * (mesh[2] - mesh[1]) / (2π)
	
	(model=model, mesh=mesh, β=β, ρc=ρc, ργ=ργ)
end;

# ╔═╡ c3ea9353-5b43-44d9-a491-a54f62e40aba
begin
	ωs = nca.mesh
	ts = ft(ωs, ωs; mode= -1)[1]
	Θt = (t -> t > 0 ? 1 : t == 0 ? 0.5 : 0.0).(ts) # Heaviside magic
	
	GcL = ft(ωs, +im / (2π) * nca.ρc .* nf.(+ωs, nca.β); mode = -1)[2]
	GcG = ft(ωs, -im / (2π) * nca.ρc .* nf.(-ωs, nca.β); mode = -1)[2]
	
	GγL = ft(ωs, -im / (2π) * nca.ργ .* nb.(+ωs, 5e-3nca.β); mode = -1)[2]
	GγG = ft(ωs, +im / (2π) * nca.ργ .* nb.(-ωs, 5e-3nca.β); mode = -1)[2]
end;

# ╔═╡ 6c086e1e-aa38-47f1-9fe5-20f8b3fa0fdb
begin
	nff(ω, β, α=5e-2, ϵ=1.0) = ((1-ϵ) * nf(ω, β) + ϵ * nf(ω + α, β)) + ϵ * (nf(-ω, β) - nf(-ω + α, β))
	
	plot(ωs, nff.(ωs, nca.β, 5e-2, 1e-1); xlims=(-1,1))
end

# ╔═╡ 717f4e9d-a037-4f2b-b6bf-38c537c841f3
begin
	""" ~: A(τ) –> A(-τ) """
	~(a) = -conj(a)
	
	""" Calculates GR(ω) and G<(ω) """
	function calculate_Gω(ΣR, ΣL)
		GR = zero(ΣR)
		@. GR[:, 1] = ωs + nca.model.ε₀f - ΣR[:, 1]
		@. GR[:, 2] = ωs - ΣR[:, 2]
		@. GR = inv(GR)
		
		GL = GR .* ΣL .* conj(GR)
		
		# We normalise it by Q because the fixed-point of G<(ω) is scale invariant
		# f(α G<) = α G<,  ∀α
		GL .*= (2π) / (sum(imag(-GL[:,1] + 2GL[:,2])) * (ωs[2] - ωs[1]))
		
		return GR, GL
	end
	
	""" Calculates Σ>(t) and Σ<(t) """
	function calculate_Σt(GG, GL)
		V2G = nca.model.V₀^2 .+ 1.0im * nca.model.Γ₀^2 * (GγG + ~GγL)
		V2L = nca.model.V₀^2 .+ 1.0im * nca.model.Γ₀^2 * (GγL + ~GγG)
		
		Σt = zeros(ComplexF64, length(ωs), 4)
		@. Σt[:,1] = -2.0im * V2G * GG[:,2] * ~GcL
		@. Σt[:,2] = +1.0im * V2G * GG[:,1] * GcG
		@. Σt[:,3] = -2.0im * V2L * GL[:,2] * ~GcG
		@. Σt[:,4] = +1.0im * V2L * GL[:,1] * GcL
		
		return Σt
	end
	
	function loop(Σt)
		# Σ>(t) –> ΣR(ω) & Σ<(t) –> Σ<(ω)
		ΣR = mapslices(x -> ft(ts, x .* Θt; mode = +1)[2], Σt[:,1:2]; dims = 1)
		ΣL = mapslices(x -> ft(ts, x; mode = +1)[2], Σt[:,3:4]; dims = 1)	

		GR, GL = calculate_Gω(ΣR, ΣL)

		# GR(ω) –> G>(t) & G<(ω) –> G<(t)
		GG = mapslices(x -> ft(ωs, 1im/1π * imag(x); mode = -1)[2], GR; dims=1)
		GL = mapslices(x -> ft(ωs, 1im/2π * imag(x); mode = -1)[2], GL; dims=1)

		Σt′ = calculate_Σt(GG, GL)
	end
end;

# ╔═╡ 914c7781-617a-4b5f-b404-ebd8f60369e4
# Solving
begin	
	Σs = mapslices(hcat(-nca.ρc, -nca.ρc, -nca.ρc, +nca.ρc), dims=1) do x
            ft(ωs, +1im / 2π * x; mode=-1)[2]
        end
	
	@time begin
		Σs = nlsolve(x -> loop(x) - x, Σs; 
			method=:anderson, show_trace=true, fatol=1e-2)
		Σs = nlsolve(x -> loop(x) - x, Σs; 
			method=:krylov, show_trace=true, fatol=1e-7, iterations=10)
	end
end;

# ╔═╡ 2f3d3871-3c84-49de-a0e9-c58a6d66ded2
# Projection
begin
	ΣR = mapslices(x -> ft(ts, x .* Θt; mode = +1)[2], Σs[:,1:2]; dims = 1)
	ΣL = mapslices(x -> ft(ts, x; mode = +1)[2], Σs[:,3:4]; dims = 1)
	
	GR, GL = calculate_Gω(ΣR, ΣL)
	
	GG = mapslices(x -> ft(ωs, 1im / 1π * imag(x); mode = -1)[2], GR; dims = 1)
	GL = mapslices(x -> ft(ωs, 1im / 2π * imag(x); mode = -1)[2], GL; dims = 1)
	
	Gd = ft(ts, (@. +1.0im * (GG[:,2] * ~GL[:,1] - 2GL[:,2] * ~GG[:,1]) * Θt); mode=+1)[2]
end;

# ╔═╡ 8b99f831-2338-4df0-b1bb-d3ea48311115
begin
	plot(ωs, -imag(Gd))
end

# ╔═╡ Cell order:
# ╟─9206bdaa-2cf6-11ec-17c9-1b978be94cdb
# ╠═d271c87b-f68b-4ae5-a1d3-576207ed7bbb
# ╠═01e9c802-cbb0-409f-ba23-ef8510c59b1c
# ╠═6c086e1e-aa38-47f1-9fe5-20f8b3fa0fdb
# ╠═c3ea9353-5b43-44d9-a491-a54f62e40aba
# ╠═717f4e9d-a037-4f2b-b6bf-38c537c841f3
# ╠═914c7781-617a-4b5f-b404-ebd8f60369e4
# ╠═2f3d3871-3c84-49de-a0e9-c58a6d66ded2
# ╠═8b99f831-2338-4df0-b1bb-d3ea48311115
