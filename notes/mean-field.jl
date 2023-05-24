### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ 22f5d22d-c092-4b44-a2da-af78f1978c8a
begin
	using Plots, LaTeXStrings

	Plots.gr()

	# Plots.default()
	Plots.default(
		fontfamily = "Computer Modern",
		background_color_legend = nothing,
		foreground_color_legend = nothing,
		tick_direction = :in,
		minorticks = true,
		framestyle = :box,
		size = (600, 450),
		titlefontsize = 16,
		guidefontsize = 11,
		legendfontsize = 9,
		tickfontsize = 9,
		grid = true,
		)

	my_colors = Dict(
		:c => Plots.Colors.colorant"#0C5DA5", 
		:dω => Plots.Colors.colorant"#FF9500",
		:dτ => Plots.Colors.colorant"#00B945",
		:ωK => Plots.Colors.colorant"#9e9e9e",
		:εd => Plots.Colors.colorant"#845B97",
		:ω0 => Plots.Colors.colorant"#FF2C00"
		)
end;

# ╔═╡ a9e98579-eaa0-45f5-8353-065e05a3e093
function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(
        m,
        Expr(
            :toplevel,
            :(eval(x) = $(Expr(:core, :eval))($name, x)),
            :(include(x) = $(Expr(:top, :include))($name, x)),
            :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
            :(include($path)),
        ),
    )
    m
end;

# ╔═╡ f2b6c08c-29ed-11eb-2363-6143e3b233a5
begin
	using Pkg; Pkg.activate()

	using StaticArrays
	using LinearAlgebra # GenericLinearAlgebra, 

	using ForwardDiff

	import NLsolve
	nlsolve(f, x0; ftol=1e-12) = NLsolve.nlsolve(f, x0; 
		method=:newton, 
		autodiff=:forward, 
		ftol=ftol, 
		iterations=20,
		inplace=false, 
		show_trace=true)

	const EchoPulse = ingredients("../src/EchoPulse.jl")
	
	using DifferentialEquations
	using Serialization
end

# ╔═╡ 25366804-7db3-4672-8c80-2d762e6b6675
begin
	# using HCubature: hcubature
	# k-integral
	# ∫dk(f; dim = 3) = hcubature(f, repeat([-π], dim), repeat([π], dim); rtol=1e-3, atol=1e-6)[1] / (2π)^dim

	bethe_dos(ε) = 1 / pi * sqrt(1 - (ε / 2)^2)
	εs = range(-2.0, 2.0, length=2^9)
	∫dε(f) = sum(f(ε) * bethe_dos(ε) for ε in εs) * (εs[2] - εs[1])
	∫dεi(f) = sum(f(i) * bethe_dos(ε) for (i, ε) in enumerate(εs)) * (εs[2] - εs[1])
end

# ╔═╡ ba6ea9f1-2d8f-4853-b433-9825aef236a9
begin
	# """
	# Cubic Anderson model parameters (units of tc)
	# """
	# Base.@kwdef mutable struct PAM
	# 	V0::Float64         # Hybridization strength
	# 	tf::Float64         # F-electron hopping
	# 	ε0f::Float64        # F-electron ground state e
	# 	μ::Float64 = 0.0    # Chemical potential

	# 	Γ0::Float64 = 0.0   # Photon hybridization strength
	# 	ω0::Float64 = 0.0   # Photon frequency
	# 	Ω0::Float64 = 0.0   # Photon width
	# 	tγ::Float64 = 0.0   # Photon incidence time
	# 	Nγ::Float64 = 0.0   # Photon pulse occupation
	# end

	"""
	Aux-boson mean-field Hamiltonian
	"""
	function H(k::SVector, p, b, λ)
		c = -2.0 * sum(cos, k) - p.μ
		f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
		h = p.V0 * b
		return @SMatrix [c h'; h f]
	end

	function H(ε::Float64, p, b, λ)
		c = ε #- p.μ
		f = λ #- p.μ
		# f = p.tf * abs2(b) * ε + λ - p.μ
		h = p.V0 * b
		return @SMatrix [c h'; h f]
	end

	"""
	Effective action at T -> 0
	"""
	function Seff(x::Vector, p)
		b, λ = x

		# The factor of 2 accounts for spin-degeneracy
		return (λ - p.ε0f) * (abs2(b) - 1) + 2∫dε(ε -> begin
			εs = eigvals(Hermitian(H(ε, p, b, λ)))
			sum(εi for εi in εs if εi < λ)
			end
		)
	end

	"""
	Saddle-point equations at T -> 0
	"""
	function saddle_points(x::Vector, p)
		return ForwardDiff.gradient(y -> Seff(y, p), x)
	end
end;

# ╔═╡ 31f97be0-2a6e-11eb-3f71-1f370f449c07
begin
	"""
	Auxiliary-boson saddle-point occupation of the fermionic fields
	"""
	function GL0(ε::Float64, p, b::Number, λ::Number)
	    c = ε #- p.μ
		f = λ #- p.μ
	    # f = p.tf * abs2(b) * ε + λ - p.μ
	    h = (p.V0 * b)

	    σ = 0.5 * (c + f)
	    δ = 0.5 * (c - f)
	    Δ = sqrt(δ^2 + abs2(h))
	    # return +im * @SMatrix [1 - δ/Δ -h'/Δ; -h/Δ 1 + δ/Δ]
		return +im * [1 - δ/Δ -h'/Δ; -h/Δ 1 + δ/Δ]
	end
	function GL0(k::SVector, p, b::Number, λ::Number)
	    c = -2.0 * sum(cos, k) - p.μ
	    f = -2.0p.tf * sum(cos, k) * abs2(b) + λ - p.μ
	    h = (p.V0 * b)

	    σ = 0.5 * (c + f)
	    δ = 0.5 * (c - f)
	    Δ = sqrt(δ^2 + abs2(h))
	    return +im * @SMatrix [1 - δ/Δ -h'/Δ; -h/Δ 1 + δ/Δ]
	end
end;

# ╔═╡ e7d8ca15-dd35-4505-b39a-17325d4e7762
begin
	function _solve(pam; tmax, kwargs...)	
	    """
	    Non-equilibrium saddle-point equations
	    """
	    function sp_neq(du, u, _, t)
			λ = u[1]
	        b = u[2] + im * u[3]
			
			G = reshape(reinterpret(ComplexF64, u[4:end]), 2, 2, :)
			
	        n = ∫dεi(i -> begin
	            g = G[:, :, i]
	
	            α = g[2,1] * b + g[1,2] * b' 
	            dαdb = g[1,2] + pam.V0 * (g[1,1] * g[2,2] * b + g[1,2] * g[1,2] * b')
	            dαdλ = g[2,2] * α
	
	            @SVector [g[1,2], g[2,2], α, dαdb, dαdλ]
	        end)

			gγ = t * EchoPulse.GγL0_ext(t, t; model = pam)

			dλ = 1 - abs2(b) + im * n[2]
	        dλ+= 2pam.Γ0^2 * gγ * n[3] * n[5]

	        db = -im * ((λ - pam.ε0f) * b - im * (pam.V0 * n[1]))
	    	db+= -im * 2pam.Γ0^2 * gγ * n[3] * n[4]

			du[1] = real(dλ)
			du[2] = real(db)
			du[3] = imag(db)
			
			_dG = @views reinterpret(ComplexF64, du[4:end])
			dG = reshape(_dG, 2, 2, length(εs))

			for (i, ε) in enumerate(εs)
				dG[:,:,i] .= (o = -1.0im * H(ε, pam, b, λ) * G[:,:,i]; o - o')
			end

			return du
	    end
	
		# Find the fields that minimize the Free Energy
		# TODO: solve equations with a complex b (composed of two real components)
		# For some reason the time-evolution is unstable when imag(b) > 0
		@time sol = nlsolve(x -> saddle_points(x, pam), [2.5e-1, 2.0e-1])
		@show sol
		
	    # Initial condition
		u1 = [real(sol.zero[2]), real(sol.zero[1]), imag(sol.zero[1])]
	    u2 = vcat([GL0(ε, pam, sol.zero[1], sol.zero[2])[:] for ε in εs]...)
		u2 = reinterpret(Float64, u2)
		u0 = [u1; u2]
		@assert eltype(u0) <: Real
		
		d = ones(Int64, length(u0))
		d[1] = 0
		
	    prob = ODEProblem(
			ODEFunction(sp_neq, mass_matrix = Diagonal(d)),
			u0, 
			(0.0, tmax)
		)
		
	    sol_t = solve(prob,
			ImplicitEuler(; autodiff = false);
			callback = FunctionCallingCallback((u, t, _) -> (@show t)),
			kwargs...
		)
	end
end;

# ╔═╡ 0bc43e51-0c4f-4af5-a526-00e2d99ccd29
# let
# 	str = "sp0"
	
# 	pams = []
# 	sols = []
	
# 	for Nγ in [1e-1, 5e-1, 1e0, 2e0, 4e0, 6e0, 8e0][[3]]
# 		for V0 in sqrt.([0.0925, 0.1084, 0.1243, 0.1561, 0.2038, 0.2674, 0.3628, 0.49])[[1:2]]
# 			for ε0f in [-0.2, -0.25, -0.3, -0.35, -0.4, -0.45][[4]]
# 				pam = EchoPulse.PhotonAssistedModel(
# 					V0 = V0, 
# 					ε0f = ε0f,
# 					Γ0 = sqrt(0.002),
# 					ω0 = 0.0,
# 					Ω0 = 0.1 * 5,
# 					tγ = 100.0 / 5,
# 					Nγ = Nγ
# 				)
# 				push!(sols, _solve(pam; 
# 					tmax = 300 / 5,
# 					dtmax = 0.5 / 5,
# 					reltol = 5e-5,
# 					abstol = 5e-5)
# 				)
# 				push!(pams, pam)
# 			end
# 		end
# 	end

# 	serialize("../data/$(str).jls", (
# 		sols= [(t = s.t, u = s.u) for s in sols], 
# 		pams = [Dict(key=>getfield(p, key) for key ∈ fieldnames(EchoPulse.PhotonAssistedModel)) for p in pams])
# 	)
# end

# ╔═╡ 60e7da8a-75d4-4644-9574-a61fda0b3d9e
begin
	str = "sp0"
	
	(sols, pams) = deserialize("../data/$(str).jls")
	tmax = sols[1].t[end]
end;

# ╔═╡ 720e196f-78e9-4c92-ae62-3568951dc45e
let
	colours = let
	  colourmap = cgrad(:viridis)
	  [colourmap[i] for i in reverse(range(0.0, 0.6, length=length(sols)))]
	end;
	
	ylims = (0.77, 1.01)
	xlims = (0, tmax)

	p1 = plot( 
		xlabel = L"T\ t", 
		# ylabel = L"\lambda_0(t)",# / \lambda(t_0)", 
		ylabel = L"\lambda_0(t) / \operatorname{max}\,\lambda_0(t)", 
		ylims = ylims,
		xlims = xlims,
	)
	
	p1 = plot!(twinx(),
		sols[1].t, 
		t -> -imag(EchoPulse.:GγL0_ext(t, t; model=(; pams[1]...))),
		xlims = xlims,
		ylims = (0.0, pams[1][:Nγ]),
		fill = (0, :grey, 0.5), 
		line = nothing, 
		legend = nothing, 
		yticks = nothing,
		xticks = nothing,
	)

	for i in reverse(collect(eachindex(sols)))
		plot!([Inf], [Inf];
			line = (1.0, colours[i]), 
			label=L"V_0 = %$(round(pams[i][:V0], digits=2))",
			# label=L"n_a = %$(round(pams[i][:Nγ], digits=2))",
			legend = :bottomright
			)
	end
	for (i, sol) in enumerate(sols)
		
		# Normalise data
		# data = (u -> sqrt(u[2]^2 + u[3]^2)).(sol.u)
		data = (u -> u[1]).(sol.u)
		data ./= maximum(data[10:end])
		
		p1 = plot!(twinx(), 
			sol.t, data, 
			label = nothing,
			line = (1.0, colours[i]), 
			ylims = ylims,
			xlims = xlims,
			xticks = nothing,
			yticks=([data[1], ], [round(sol.u[1][1], digits=2), ]), 
			ymirror=true,
			ylabel = isone(i) ? L"\lambda_0(t_0)" : ""
		)

		hline!([data[1],], line = (:dash, 1.0, colours[i], 0.4), label=nothing)
	end

	# p1 = plot!()
	savefig("$(str).pdf")
end

# ╔═╡ Cell order:
# ╟─a9e98579-eaa0-45f5-8353-065e05a3e093
# ╠═f2b6c08c-29ed-11eb-2363-6143e3b233a5
# ╠═22f5d22d-c092-4b44-a2da-af78f1978c8a
# ╠═25366804-7db3-4672-8c80-2d762e6b6675
# ╠═ba6ea9f1-2d8f-4853-b433-9825aef236a9
# ╠═31f97be0-2a6e-11eb-3f71-1f370f449c07
# ╠═e7d8ca15-dd35-4505-b39a-17325d4e7762
# ╠═0bc43e51-0c4f-4af5-a526-00e2d99ccd29
# ╠═60e7da8a-75d4-4644-9574-a61fda0b3d9e
# ╠═720e196f-78e9-4c92-ae62-3568951dc45e
