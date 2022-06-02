### A Pluto.jl notebook ###
# v0.19.6

using Markdown
using InteractiveUtils

# ╔═╡ c56b9d66-e0e7-11ec-324d-630a482f2ab7
begin
	import Pkg; Pkg.activate()
	
	using SpecialFunctions
	using FFTW

	function _ft(xs, ys; inverse::Bool = false)
		@assert issorted(xs)
		
		L = length(xs)
		dx = xs[2] - xs[1]
		
		# Because the FFT calculates the transform as
		#   ỹ_k = \sum_j e^{±2pi i j k/n} y_j, from j=0 to j=n-1,
		# we need to transform this into the time and frequency units, 
		# which ends up scaling the frequencies `x̂s` by (2pi / dx).
		x̂s = fftfreq(L, 2π / dx)
		
		# The resulting Fourier transform also picks up a phase
		ℯⁱᵠ = (inverse ? 1 / (2π) : 1.0) * dx * exp.((inverse ? -1.0 : 1.0) * im * xs[1] .* x̂s) 
		
		# FFT
		ŷs = ℯⁱᵠ .* (inverse ? fft(ys) : bfft(ys))
		
		return fftshift(x̂s), fftshift(ŷs)
	end
	
	ft(x...; xs...) = _ft(x...; xs...)[2]

	using LaTeXStrings, Plots
end

# ╔═╡ dd2acffb-2da8-4dc6-9634-8c1aa88ff134
begin
	using NLsolve
	using PyCall
	
	function nlsolve_(f, x0; method, kwargs...)   
	    optimize = pyimport("scipy.optimize")
	    
	    sol = optimize.root(
	        fun=x -> f(reshape(x, size(x0)...)) |> vec,
	        x0=vec(x0),
	        method=method,
	        options=kwargs
		)
	    
	    return reshape(sol["x"], size(x0)...)
	end

	function fixed_point(f, x0)
	    x0 = nlsolve(x -> f(x) - x, x0; method = :anderson,
			iterations=200, show_trace = :true, ftol = 1e-8, beta = 0.6, m=3).zero
		# x0 = nlsolve_(x -> f(x) - x, x0; method=:anderson, 
		# 	maxiter=30, disp=true, fatol=1e-2)
		# x0 = nlsolve_(x -> f(x) - x, x0; method=:krylov,
		# 	maxiter=40, disp=true, fatol=1e-9, jac_options=Dict(:method => "lgmres"))
	end
end;

# ╔═╡ dc41646c-f1a8-4dc0-a38e-3606eb4328a7
begin
	""" ~: -A(τ)^* = A(-τ) """
	~(a) = -conj(a)
	
	""" Calculates GR(ω) and G<(ω) in equilibrium """
	function calculate_Gω(ωs, model, ΣR, ΣL)
	    GR = zero(ΣR)
	    @. GR[:, 1] = ωs + model.ε0f - ΣR[:, 1]
	    @. GR[:, 2] = ωs - ΣR[:, 2]
	    @. GR[:, 3] = ωs - (model.ε0f + model.U) - ΣR[:, 3]
	    @. GR = inv(GR)
	
	    GL = GR .* ΣL .* conj(GR)
	
	    # We normalise it by Q because the fixed-point of G<(ω) is scale invariant
	    # f(α G<) = α G<,  ∀α
	    GL *= 2π / ((ωs[2] - ωs[1]) * sum(imag(-GL[:, 1] + 2GL[:, 2] - GL[:, 3])))
	
	    return GR, GL
	end
	
	""" Calculates Σ>(t) and Σ<(t) in equilibrium """
	function calculate_Σt(model, ρcG, ρcL, GG, GL)
	    V2G = model.V0^2
	    V2L = model.V0^2
	
	    Σt = zeros(ComplexF64, length(ρcL), 6)
	    @. Σt[:, 1] = -2im * V2G * GG[:, 2] * ~ρcL
	    @. Σt[:, 4] = -2im * V2L * GL[:, 2] * ~ρcG
	    @. Σt[:, 2] = +1im * V2G * (GG[:, 1] * ρcG - GG[:, 3] * ~ρcL)
	    @. Σt[:, 5] = +1im * V2L * (GL[:, 1] * ρcL - GL[:, 3] * ~ρcG)
	    @. Σt[:, 3] = +2im * V2G * GG[:, 2] * ρcG
	    @. Σt[:, 6] = +2im * V2L * GL[:, 2] * ρcL
	
	    return Σt
	end

	gaussian(x, x₀, σ) = exp(-0.5 * ((x - x₀) / (σ))^2) / sqrt(2 * π * σ^2)
	
	nf(ω, β) = 0.5 * (1.0 - tanh(0.5 * β * ω))

	function symmetrize(x)
		n = length(x) ÷ 2
		[x[1:n+1]; ~reverse(x[1:n])]
	end
end;

# ╔═╡ 3ba9a6a3-0f7b-42f5-918c-3533d3913984
# function dmft_loop(Σt) # Bethe lattice
	
# 	# Calculate G>(t) and G<(t)
# 	GG, GL = let
# 		# Σ>(t) –> ΣR(ω) & Σ<(t) –> Σ<(ω)
# 		ΣR = mapslices(x -> ft(ts, x .* Θt; inverse = false), Σt[:, 1:3]; dims = 1)
# 		ΣL = mapslices(x -> ft(ts, x; inverse = false), Σt[:, 4:6]; dims = 1)
# 		GR, GL = calculate_Gω(ωs, model, ΣR, ΣL)

# 		# GR(ω) –> G>(t) & G<(ω) –> G<(t)
# 		GG = mapslices(x -> ft(ωs, im * 2imag(x); inverse = true), GR; dims = 1)
# 		GL = mapslices(x -> ft(ωs, im * 1imag(x); inverse = true), GL; dims = 1)
# 		GG, GL
# 	end

# 	# Projection
# 	GdR = ft(ts, (@. +im * ((GG[:, 2] * ~GL[:, 1] - GG[:, 3] * ~GL[:, 2]) - (GL[:, 2] * ~GG[:, 1] - GL[:, 3] * ~GG[:, 2])) * Θt); inverse = false)

# 	Gloc = let
# 		G0cR = ft(ts, Σt[:, 7] .* Θt_ ; inverse = false)
# 		@. G0cR + abs2(model.V0) * G0cR * GdR * G0cR
# 	end
	
# 	G0cR = @. inv(ωs - Gloc)

# 	G0cR_ = ft(ωs, 2im * imag(G0cR); inverse = true) .* Θt

# 	# Calculate new self-enegies
# 	Σt′ = let
# 		GcG = ft(ωs, +2im * imag(G0cR) .* nf.(-ωs, β); inverse = true)
# 		GcL = ft(ωs, -2im * imag(G0cR) .* nf.(+ωs, β); inverse = true)
		
# 		calculate_Σt(model, GcG, GcL, GG, GL) |> x -> mapslices(symmetrize, x; dims=1)
# 	end
	
# 	hcat(Σt′, G0cR_), (; Gloc, GdR, G0cR)
# end;

# ╔═╡ ac5276df-dad5-49fc-a8b6-21e20ee0dc5b
begin
	β = 2e2
	model = (V0 = sqrt(0.1), ε0f = -0.4, U = Inf)
	# model = let
	# 	U = 0.75
	# 	(V0 = sqrt(0.10), ε0f = -U/2, U = U)
	# end

	# Frequency grid
	ωs = range(-3, 3, length=2^15 - 1) |> collect
	ts = _ft(ωs, ωs; inverse = true)[1] |> collect
    Θt = (t -> t > 0 ? 1.0 : t == 0 ? 0.5 : 0.0).(ts) # Heaviside magic
	Θt_= (t -> t >= 0 ? 1.0 : 0.0).(ts) # Heaviside magic
end;

# ╔═╡ 14145db6-3237-42ae-a3ed-9d50644fab31
function dmft_loop(Σt)
	# Calculate G>(t) and G<(t)
	GG, GL = let
		# Σ>(t) –> ΣR(ω) & Σ<(t) –> Σ<(ω)
		ΣR = mapslices(x -> ft(ts, x .* Θt; inverse = false), Σt[:, 1:3]; dims = 1)
		ΣL = mapslices(x -> ft(ts, x; inverse = false), Σt[:, 4:6]; dims = 1)
		GR, GL = calculate_Gω(ωs, model, ΣR, ΣL)

		# GR(ω) –> G>(t) & G<(ω) –> G<(t)
		GG = mapslices(x -> ft(ωs, im * 2imag(x); inverse = true), GR; dims = 1)
		GL = mapslices(x -> ft(ωs, im * 1imag(x); inverse = true), GL; dims = 1)
		GG, GL
	end

	# Projection
	GdR = ft(ts, (@. +im * ((GG[:, 2] * ~GL[:, 1] - GG[:, 3] * ~GL[:, 2]) - (GL[:, 2] * ~GG[:, 1] - GL[:, 3] * ~GG[:, 2])) * Θt); inverse = false)
	
	# Calculate the Weiss field
	G0cR, Gloc = let
		# ΣcR(t) -> ΣcR(ω)
		# Θt is added here because the solver doesn't respect that symmetry
		ΣcR = ft(ts, Σt[:, 7] .* Θt_; inverse = false)
		# ΣcR[imag(ΣcR) .> 0.0] .= real(ΣcR[imag(ΣcR) .> 0.0]) .- 1e-14im
		
		# Calculate the lattice sum
		# Gloc = let # gaussian
		# 	x = ωs - ΣcR
		# 	exp.(-0.5 * x.^2) .* sqrt(pi/2) .* (-im .+ erfi.(x./sqrt(2)))
		# end
		# Gloc = let # square
		# 	x = ωs - ΣcR
		# 	acoth.(x)
		# end
		Gloc = let # bethe
			x = ωs - ΣcR
			@. -1/2 *(-1 + sqrt(1 - 4/x^2)) * x
			# @. -2/1 *(-1 + sqrt(1 - 1/x^2)) * x
		end
		
		(@. inv(inv(Gloc) + ΣcR)), Gloc
	end
	# # enforces Kramers-Kronig
	# G0cR = ft(ts, ft(ωs, 2im * imag(G0cR); inverse = true) .* Θt, inverse = false)
	
	# Calculate new self-enegies
	Σt′ = let
		GcG = ft(ωs, +2im * imag(G0cR) .* nf.(-ωs, β); inverse = true)
		GcL = ft(ωs, -2im * imag(G0cR) .* nf.(+ωs, β); inverse = true)
		
		calculate_Σt(model, GcG, GcL, GG, GL) |> x -> mapslices(symmetrize, x; dims=1)
	end

	# Calculate Weiss self-energy
	ΣcR = (@. (inv(inv(GdR)/abs2(model.V0) + G0cR)))
	ΣcR = ft(ωs, 2im * imag(ΣcR); inverse = true) .* Θt #2im * imag(ΣcR) would be perfect but it doesn't always converge
	
	hcat(Σt′, ΣcR), (; Gloc, GdR, G0cR)
end;

# ╔═╡ 2f8fc3ee-9485-472e-9022-b8e504c7b1ff
begin
	# Initial condition
	ρc = ft(ωs, 2im * (-π * gaussian.(ωs, 0.0, 1.0)); inverse = true)
	
	# Solve NCA equations
	Σt = fixed_point(x -> dmft_loop(x)[1], hcat(ρc, ρc, ρc, -ρc, ρc, ρc, ρc .* Θt))
end;

# ╔═╡ 0ab44553-8e84-4012-ad71-0cc1f9555e6d
begin
	(; Gloc, GdR, G0cR) = dmft_loop(Σt)[2];
	plot(ωs, -Gloc |> imag)
	# plot!(ωs, -(@. G0cR + G0cR * abs2(model.V0) * GdR * G0cR) |> imag)
	# plot!(ωs, -G0cR |> imag)
	# plot!(ωs, -GdR |> imag)
	# xlims!(-0.1,0.1)
end

# ╔═╡ Cell order:
# ╠═c56b9d66-e0e7-11ec-324d-630a482f2ab7
# ╠═dd2acffb-2da8-4dc6-9634-8c1aa88ff134
# ╠═dc41646c-f1a8-4dc0-a38e-3606eb4328a7
# ╠═14145db6-3237-42ae-a3ed-9d50644fab31
# ╠═3ba9a6a3-0f7b-42f5-918c-3533d3913984
# ╠═ac5276df-dad5-49fc-a8b6-21e20ee0dc5b
# ╠═2f8fc3ee-9485-472e-9022-b8e504c7b1ff
# ╠═0ab44553-8e84-4012-ad71-0cc1f9555e6d
