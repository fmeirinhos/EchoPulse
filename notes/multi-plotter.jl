### A Pluto.jl notebook ###
# v0.19.23

using Markdown
using InteractiveUtils

# ╔═╡ aef331b5-c5b3-4eb2-8713-278904861477
begin
	using LsqFit
end

# ╔═╡ 9c40c777-cdd0-4436-ba63-858226cc050d
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

# ╔═╡ 1e759b6c-8b76-11ed-327f-456a27dc60d5
begin
    using Pkg
    Pkg.activate()
	
	const EchoPulse = ingredients("../src/EchoPulse.jl")

	using Serialization

	using Plots, LaTeXStrings
end; 

# ╔═╡ 40da4bf1-97b5-4b7a-a47c-495e425e40f5
begin
    Plots.gr(); Plots.default()
	
    Plots.default(
        fontfamily = "Computer Modern",
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        tick_direction = :in,
        minorticks = true,
        framestyle = :box,
        size = 2.0 * 288.00 .* ((1 + sqrt(5))/2, 1),
        titlefontsize = 18,
		guidefontsize = 11,
		legendfontsize = 9,
		tickfontsize = 9,
        grid = false,
		margin = 5Plots.mm
    )

	Plots.scalefontsizes(2.0)

    my_colors = Dict(
        :c => Plots.Colors.colorant"#0C5DA5",
        :dω => Plots.Colors.colorant"#FF9500",
        :dτ => Plots.Colors.colorant"#00B945",
        :ωK => Plots.Colors.colorant"#9e9e9e",
        :εd => Plots.Colors.colorant"#845B97",
        :ω0 => Plots.Colors.colorant"#FF2C00",
    )
end;

# ╔═╡ fdf9c03e-d8fc-4218-9c46-7d0b165811f5
begin
	Base.adjoint(x::LaTeXStrings.LaTeXString) = x
	Base.adjoint(x::Symbol) = x
end

# ╔═╡ 22c04e92-eaa8-4ca2-8388-9ba24db1672f
begin
	str = "rrun"
	list_j = [-0.35, -0.39]
	list_k = [1.0, 3.0, 5.0, 7.0, 10.]

	# str = "run-run-run"
	# list_j = [5.0e-3, 1.0e-3, 5.0e-2, 1e-1, 5.0e-1, 1e-4, 5e-4] # ω0
	# list_k = [6.0e-2, 9.0e-2, 2e-1] # Ω0
	
	# str = "new-run"
	# list_j = [nothing] # ε0f
	# list_k = [1e0, 3e0, 7e0, 1e1, 1.5e1, 3e1, 5e1, 8.5e1, 1e2, 1.5e2, 1.9e2]

	# str = "run-run-run-run"
	# list_j = [-0.35, -0.39]
	# list_k = [sqrt(0.02),  sqrt(0.03), sqrt(0.04), sqrt(0.05)]
end;

# ╔═╡ 20dcd8bd-042c-4340-9ddd-929e5e3b64f4
begin
	_Ts = []
	_ωs = []
	_params = []
	_nγ0 = []
	_nγ = []
	_occupied_states = []
	_kondo_maximum_spectral = []

	for j in eachindex(list_j)
		for k in eachindex(list_k)
			r = (k-1) + length(list_k) * (j-1)

			(; Ts, ωs, data, params) = deserialize("../data/$(str)-$(r).jls")
			(; ρd_ω, GdL_ω, ρc_ω, GcL_ω, ργ_ω, GγL_ω, GγL_vac, ΣcR_ω) = data

			# t0 index
			i_t0 = findmin(abs, Ts)[2]
		
			# Integral over all frequencies
			∫dω(gf) = sum(gf, dims=1) * (ωs[2] - ωs[1]) / (2π) |> x -> dropdims(x, dims=1)
		
			# Kondo points
			w = 3
			i1 = argmin(abs.(ωs .+ w * params.ωK))
			i2 = argmin(abs.(ωs .- w * params.ωK))
			
			# Integral over Kondo frequencies
			∫dωK(gf) = sum(gf[i,:] for i in i1:i2) * (ωs[2] - ωs[1]) / (2π)
			
			# Calculate photon number
			nγ0 = -imag([EchoPulse.GγL0_ext(t,t; model=params) for t in Ts])
			nγ = -imag(∫dω(GγL_ω))

			# Calculate hole number
			push!(_occupied_states, -imag(∫dω(ρd_ω + ρc_ω + GdL_ω + GcL_ω)))
			
			push!(_Ts, Ts)
			push!(_params, params)
			push!(_nγ0, nγ0)
			push!(_nγ, nγ)
			push!(_kondo_maximum_spectral, map(x -> maximum(-imag(x[i1:i2])), eachcol(ρd_ω)))
		end
	end
	
	GC.gc()
	varinfo(@__MODULE__, imported = true)
end;

# ╔═╡ ce2c9fca-1345-457e-b43b-5c11ff8f14db
let
	@assert str == "new-run"

	r = eachindex(list_k)
	
	colours = let
		colourmap = cgrad(:viridis)
		grid = (log.(list_k[r]) ./ log(last(list_k[r]))) * 0.9
		[colourmap[i] for i in reverse(grid)]
	end

	xlims = (100, 600)
	ylims = (-0.000, 0.004)

	p1 = plot(
		_Ts[r[1]], 
		_nγ0[r[1]] * (1/3000),
		line = (:black, 2.0),
		fill = (0, :grey, 0.5), 
		label = nothing,
		legend = :topleft,
		xlabel = L"t / \tau_K",
		ylabel = L"-\operatorname{Im}\ G^<_a(t, t)",
		xlims = xlims,
		ylims = ylims,
		xtick= _params[1].tγ .- (-30:1:30) .* _params[1].τK,
		xformatter = x -> (x - _params[1].tγ) / _params[1].τK,
	)

	plot!([Inf], [Inf], color=:white, label=L"T/D")
	
	i_max = [findmin(x -> abs(x - 600), _Ts[i])[2] for i in eachindex(_Ts)]

	# Scale the x-axis by a factor of 2 because the half-bandwidth in a Bethe lattice is 2υ
	scale = 2
	
	plot!(
		_Ts[r], 
		[_nγ[i] .- _nγ[i][i_max[i]] for i in r],
		label = [L"1/%$(Int(param.β * scale))" for param in _params[r]]',
		legendfontsize = 14,
		line = (3.0, colours')
	)

	vline!([_params[end].tγ + _params[end].τK], 
		line = (2.0, :black, :dot, 0.5),
		label = nothing
	)
	
	int_max = let
		nγ = [_nγ[i] ./ _nγ[i][i_max[i]] .- 1 for i in r]
		# nγ = [_nγ[i] .- _nγ[i][i_max[i]] for i in r]
		[
			maximum(nγ[1][3500:3600]), 
			maximum(nγ[2][3500:3600]), 
			maximum(nγ[3][3500:3600]), 
			maximum(nγ[4][3500:3600]), 
			maximum(nγ[5][3500:3600]), 
			maximum(nγ[6][3500:3600]), 
			maximum(nγ[7][3500:3600]),
			maximum(nγ[8][3500:3600]),
			maximum(nγ[9][3500:3600]),
			maximum(nγ[10][3500:3600]),
			maximum(nγ[11][4550:4650]),
			]
	end

	# Inset
	
	plot!(; inset = (1, bbox(0.58, 0.16, 0.4, 0.5)))

	plot!([inv(param.β) / scale for param in _params[r]], int_max, 
		xaxis=:log, 
		label = nothing,
		line = (2.0, :black, :solid, 0.5),
		subplot = 2,
		ylims = (-0.05, 1.0)
	)
	vline!([_params[end].ωK / scale], 
		subplot = 2, 
		xaxis=:log, 
		line = (2.0, :black, :dot, 0.5),
		label = nothing
	)
	annotate!(_params[end].ωK * (1 + 0.6) / scale, 0.15, text(L"T_K", 18),)
	scatter!([inv(param.β) / scale for param in _params[r]], int_max, 
		xaxis=:log, 
		xlabel = L"T / D",
		title = "Delayed peak normalized intensity", 
		titlefontsize = 14,
		label = nothing,
		marker = (6.0, colours),
		subplot = 2,
	)
	
	vspan!(
        [600, xlims[end]];
        label = nothing,
        fillcolor = :red,
        alpha = 0.1,
        linewidth = 0,
        markersize = 0,
		fillstyle = :/,
		linecolor = :black
    )
				
	# vline!([_Ts[r[1]][3500],], color=:black)
	# vline!([_Ts[r[1]][3600],], color=:black)
	# vline!([_Ts[r[2]][3500],], color=:black)
	# vline!([_Ts[r[2]][3600],], color=:black)
	# vline!([_Ts[r[3]][3500],], color=:black)
	# vline!([_Ts[r[3]][3600],], color=:black)
	# vline!([_Ts[r[4]][3500],], color=:black)
	# vline!([_Ts[r[4]][3600],], color=:black)
	# vline!([_Ts[r[5]][3500],], color=:black)
	# vline!([_Ts[r[5]][3600],], color=:black)
	# vline!([_Ts[r[6]][3500],], color=:black)
	# vline!([_Ts[r[6]][3600],], color=:black)
	# vline!([_Ts[r[7]][3500],], color=:black)
	# vline!([_Ts[r[7]][3600],], color=:black)
	# vline!([_Ts[r[8]][3500],], color=:black)
	# vline!([_Ts[r[8]][3600],], color=:black)
	# vline!([_Ts[r[9]][3500],], color=:black)
	# vline!([_Ts[r[9]][3600],], color=:black)
	# vline!([_Ts[r[10]][4550],], color=:black)
	# vline!([_Ts[r[10]][4650],], color=:black)
	# vline!([_Ts[r[11]][3500],], color=:black)
	# vline!([_Ts[r[11]][3600],], color=:black)

	# savefig("intensity-1.svg")
end

# ╔═╡ 8af64e39-b83c-48d0-aea9-cd90ddcec879
let
	@assert str == "rrun"
	# r = reshape(eachindex(_Ts), length(list_k), length(list_j))
	r = reshape(eachindex(_Ts)[1:5], length(list_k), 1)

	colours = let
	  colourmap = cgrad(:viridis)
	  [colourmap[i] for i in reverse(range(0.0, 0.7, length=size(r,1)))]
	end
	
	r = r[:]

	styles = [i <= length(list_k) ? :solid : :dash for i in r]

	xlims = (100, 600)
	ylims = (0.000, 0.004)

	p1 = plot(
		# _Ts[r[1]], 
		# _nγ0[r[1]] * (1/3000),
		line = (:black, 2.0),
		fill = (0, :grey, 0.5), 
		label = nothing,
		legend = :topleft,
		xlabel = L"t / \tau_K",
		ylabel = L"-\operatorname{Im}\ G^<_a(t, t)",
		xlims = xlims,
		ylims = ylims,
		xtick= _params[1].tγ .- (-30:1:30) .* _params[1].τK,
		xformatter = x -> (x - _params[1].tγ) / _params[1].τK,
	)

	plot!([Inf], [Inf], color=:white, label=L"\bar{n}_a")
	plot!([[Inf] for _ in eachindex(list_k)], [[Inf] for _ in eachindex(list_k)], 
		ylims = ylims,
		label = [L"%$(Int(param.Nγ))" for param in _params[eachindex(list_k)]]',
		line = (2.0, styles', colours'),
	)

	i_max = [findmin(x -> abs(x - 600), _Ts[i])[2] for i in eachindex(_Ts)]

	plot!(
		_Ts[r], 
		[_nγ[i] .- _nγ[i][i_max[i]] for i in r], 
		ylims = ylims,
		xlims = xlims,
		label = nothing,
		line = (3.0, styles', colours')
	)

	vline!([_params[1].tγ + _params[1].τK], 
		line = (2.0, :black, :dot, 0.5),
		label = nothing
	)


	m1(x, p) = @. 0p[4] + p[1] / cosh(2π * (x - p[2] - _params[1].tγ) / p[3])^2

	xs = _Ts[1]
	ys = _nγ[5] .- _nγ[5][i_max[5]]
    fit1 = curve_fit(m1, xs[3500:3580], ys[3500:3580], [1.0, 50, 300, 0.0])
    @assert fit1.converged
	@info fit1
	@info "p2 = $(fit1.param[2] / _params[5].τK) τK"
	@info "p3 = $(fit1.param[3] / _params[5].τK) τK"
	
	plot!([Inf], [Inf], color=:white, label=L"\ ")
	
	plot!(xs[3100:4000], x -> m1(x, fit1.param), 
		line = (1.5, :black, :dash),
		label = L"f(t)",
	)
	
	# plot!([Inf], [Inf], color=:white, label=L"\ ")
	# plot!([Inf], [Inf], color=:white, label=L"\varepsilon_f / D")
	# plot!([Inf], [Inf], line = (:black, :solid), label = L"%$(list_j[1] / 2)")
	# p1 = plot!([Inf], [Inf], line = (:black, :dash), label = L"%$(list_j[2] / 2)")

	_i_t0 = [findmin(abs, Ts)[2] for Ts in _Ts]
	
	_i0 = [findmin(x -> abs(x - 50), Ts)[2] for Ts in _Ts]

	# vline!([_Ts[r[1]][3500],], color=:black)
	# vline!([_Ts[r[1]][3600],], color=:black)
	# vline!([_Ts[r[2]][3500],], color=:black)
	# vline!([_Ts[r[2]][3600],], color=:black)
	# vline!([_Ts[r[3]][3500],], color=:black)
	# vline!([_Ts[r[3]][3600],], color=:black)
	# vline!([_Ts[r[4]][3500],], color=:black)
	# vline!([_Ts[r[4]][3600],], color=:black)
	# vline!([_Ts[r[5]][3500],], color=:black)
	# vline!([_Ts[r[5]][3600],], color=:black)
	# vline!([_Ts[r[6]][3150],], color=:black)
	# vline!([_Ts[r[6]][3250],], color=:black)
	# vline!([_Ts[r[7]][3150],], color=:black)
	# vline!([_Ts[r[7]][3250],], color=:black)
	# vline!([_Ts[r[8]][3150],], color=:black)
	# vline!([_Ts[r[8]][3250],], color=:black)
	# vline!([_Ts[r[9]][3150],], color=:black)
	# vline!([_Ts[r[9]][3250],], color=:black)
	# vline!([_Ts[r[10]][3150],], color=:black)
	# vline!([_Ts[r[10]][3250],], color=:black)

	int_max = let
		nγ = [_nγ[i] ./ _nγ[i][_i0[i]] .- 1 for i in r]
		# nγ = [_nγ[i] .- _nγ[i][_i0[i]] for i in r]
		[
			maximum(nγ[1][3500:3600]), 
			maximum(nγ[2][3500:3600]), 
			maximum(nγ[3][3500:3600]), 
			maximum(nγ[4][3500:3600]), 
			maximum(nγ[5][3500:3600]),
		# 	maximum(nγ[6][3150:3250]), 
		# 	maximum(nγ[7][3150:3250]), 
		# 	maximum(nγ[8][3150:3250]), 
		# 	maximum(nγ[9][3150:3250]), 
		# 	maximum(nγ[10][3150:3250])
		]
	end

	n_approx = let
		 _i_t0 = [findmin(abs, Ts)[2] for Ts in _Ts]
		km = [_occupied_states[i] .- _occupied_states[i][_i_t0[i]] for i in r]
		map(x -> maximum(x[2000:4000]), km)
	end

	# annotate!(350, (ylims[end] / 2) * 1.13, text("Delayed peak normalised intensity", 14, rotation = 90, "Computer Modern"),)
	
	plot!(; inset = (1, bbox(0.58, 0.27, 0.4, 0.5)))

	plot!(n_approx[1:5], int_max[1:5], 
		label = nothing,
		line = (2.0, :black, :solid, 0.5),
		title = "Delayed peak normalized intensity", 
		titlefontsize = 14,
		xlabel = L"N",
		subplot = 2,
		ylims = (0.0, 1.0),
		xlims = (0.0, 0.042),
		xtick = 0.00 .+ 0.01 .* (0:1:4)
	)

	scatter!(n_approx, int_max, 
		xmirror = true,
		label = nothing,
		marker = (6.0, colours),
		subplot = 2,
	)
	
	p = scatter!([Inf], [Inf], 
		label = L"N", 
		marker = (:black), 
		legend = :bottomright,
		subplot = 2,
	)
	scatter!([NaN], [NaN], 
		label = L"\bar{n}_a", 
		legend = :bottomright,
		marker = (:black, :utriangle), 
		subplot = 2,
	)

	t = twiny(p[2])
	plot!(t, [param.Nγ for param in _params[1:5]], int_max[1:5], 
		label = nothing,
		line = (2.0, :black, :solid, 0.5),
		xlabel = L"\bar{n}_a",
		xmirror = false,
		subplot = 2,
		ylims = (0.0, 0.99),
		xlims = (0.0, 10.3),
		xtick = 0 .+ (0:2:10)
	)
	scatter!(t, [param.Nγ for param in _params[r]], int_max, 
		label = nothing,
		marker = (6.0, colours, :utriangle),
		subplot = 2,
	)
	
	plot!(framestyle = :box)

	savefig("intensity-5.svg")
end

# ╔═╡ 6001df42-9cc0-483f-9578-e56e4c8ddf8f
let
	@assert str == "run-run-run-run"
	r = reshape(eachindex(_Ts), length(list_k), length(list_j))
	r = r[sortperm(list_k), :]
	
	colours = let
	  colourmap = cgrad(:viridis)
	  [colourmap[i] for i in reverse(range(0.1, 0.9, length=size(r, 1)))]
	end

	r = r[:]

	styles1 = [i <= length(list_k) ? :solid : :dash for i in r]
	styles2 = [i <= length(list_k) ? :solid : :dash for i in r]

	xlims = (0.0, 800)
	ylims = (0.0, 0.006)
	
	p1 = plot(
		_Ts[r[1]], 
		_nγ0[r[1]],
		xlims = xlims,
		ylims = ylims,
		fill = (0, :grey, 0.5), 
		label = nothing,
		line = nothing, 
		xlabel = L"T\ \upsilon", 
		ylabel = L"-\operatorname{Im}\ G^<_a(T, \tau = 0)_{\mathrm{W}}",
		margin = 5Plots.mm,
		legend = :bottomright,
		foreground_color_legend = :white,
	)

	vspan!(
        [600, xlims[end]];
        label = nothing,
        fillcolor = :red,
        alpha = 0.1,
        linewidth = 0,
        markersize = 0,
		fillstyle = :/
    )
	
	plot!([[Inf] for _ in eachindex(list_k)], [[Inf] for _ in eachindex(list_k)];
		label = [L"\alpha^2=%$(round(param.α1^2, digits=3))" for param in _params[eachindex(list_k)]]',
		line = (2.0, colours'),
	)
	
	plot!(
		_Ts[r], 
		[x .- x[end] for x in _nγ[r]], 
		xlims = xlims,
		ylims = ylims,
		label = nothing,
		# label = [L"\alpha^2=%$(round(param.α1^2, digits=3)),\ \varepsilon^f_0=%$(round(param.ε0f, digits=3))" for param in _params[r]]',
		# label = [L"\beta=%$(round(param.β, digits=3)),\varepsilon^f_0=%$(param.ε0f)" for param in _params[r]]',
		line = (2.0, styles1', colours')
	)

	plot!([Inf], [Inf], 
		line = (:black, :solid), 
		label = L"\varepsilon^f_0 = %$(list_j[1])")
	plot!([Inf], [Inf], 
		line = (:black, :dash), 
		label = L"\varepsilon^f_0 = %$(list_j[2])")
	
	_i_t0 = [findmin(abs, Ts)[2] for Ts in _Ts]

	ylims2 = (0.0, Inf)
	
	p1 = plot!(
		twinx(),
		_Ts[r], 
		[_kondo_maximum_spectral[r[i]] ./ _kondo_maximum_spectral[r[i]][_i_t0[r[i]]] for i in eachindex(r)], 
		ylims = ylims2,
		xlims = xlims,
		label = nothing,
		# label = [L"\beta=%$(round(param.α1, digits=3))" for param in _params[r]]',
		# label = [L"\beta=%$(round(param.β, digits=3)),\varepsilon^f_0=%$(param.ε0f)" for param in _params[r]]',
		line = (2.0, styles2', colours', 0.3),
		framestyle = :box,
		ylabel = "Kondo peak relative height"
	)

	plot!([740,790], [0.0055,0.0055],arrow=true,color=:black,linewidth=2,label="")
	plot!([180,10], [0.0005,0.0005],arrow=true,color=:black,linewidth=2,label="")

	# hline!(twinx(), 
	# 	[1.0,], 
	# 	line = (0.5, :dot, :black, 1.0), 
	# 	frame = nothing,
	# 	ylims = ylims2,
	# 	yticks = nothing,
	# 	xticks = nothing,
	# 	label = nothing
	# )
	
	plot!(framestyle = :box,)

	# savefig("reflectivity-2.svg")
end

# ╔═╡ dc8dc52c-3900-4bfa-a1b4-12b5f487fe0b
let
	@assert str == "run-run-run"
	r = reshape(eachindex(_Ts), length(list_k), length(list_j))
	r = r[:, sortperm(list_j)]
	r = r[1:3, [1,5]]'
	# r = r[1:3, [1,6]]'

	colours = let
	  colourmap = cgrad(:viridis)
	  [colourmap[i] for i in reverse(range(0.0, 0.9, length=size(r, 1)))]
	end

	r = r[:]

	xlims = (0.0, 800)
	ylims = (0.0, 0.004)
	
	p1 = plot(
		_Ts[r[1]], 
		_nγ0[r[1]],
		xlims = xlims,
		ylims = ylims,
		fill = (0, :grey, 0.5), 
		label = nothing,
		line = nothing, 
		xlabel = L"T\ \upsilon", 
		ylabel = L"-\operatorname{Im}\ G^<_a(T, \tau = 0)_{\mathrm{W}}",
		legend = :bottomright,
		margin = 5Plots.mm
	)
	
	plot!(
		_Ts[r], 
		[x .- x[end] for x in _nγ[r]], 
		xlims = xlims,
		ylims = ylims,
		label = [L"\omega_0=%$(round(param.ω0, digits=4))" for param in _params[r]]',
		line = (2.0, colours')
	)

	# _i_t0 = [findmin(abs, Ts)[2] for Ts in _Ts]

	# ylims2 = (0.0, Inf)

	# p1 = plot!(
	# 	twinx(),
	# 	_Ts[r], 
	# 	[_kondo_maximum_spectral[i] ./ _kondo_maximum_spectral[i][_i_t0[i]] for i in r], 
	# 	ylims = ylims2,
	# 	xlims = xlims,
	# 	label = nothing,
	# 	line = (2.0, :solid, colours', 0.3),
	# 	framestyle = :box,
	# 	ylabel = "Relative Kondo height"
	# )
	
	vspan!(
        [600, xlims[end]];
        label = nothing,
        fillcolor = :red,
        alpha = 0.1,
        linewidth = 0,
        markersize = 0,
		fillstyle = :/
    )

	plot!(; inset = (1, bbox(0.53, 0.05, 0.45, 0.45)))
	
	# model = EchoPulse.PhotonAssistedModel(
 #        # siam
 #        ;
 #        V0 = sqrt(0.09), # Hybridization strength
 #        ε0f = -0.35, # F-electron ground state energy
 #        U0 = Inf, # f-electron Coulomb repulsion

	# 	# bath
	# 	α1 = sqrt(0.05), # Coupling to the fermionic bath
 #        α2 = sqrt(10.), # Coupling to the photonic bath
	# 	cutoff = 0.25, # Photonic bath cutoff

	# 	# quantum light
 #        Γ0 = sqrt(0.002), # Photon hybridization strength
 #        ω0 = 1e-3,#2.5e-2, # Photon central frequency
 #        Ω0 = 6.0e-2,#9.0e-2, # Photon spectral width
	# 	Nγ = 1.0e1, # Photon pulse maximum average occupation
	# 	tγ = 2.5e2, # Photon pulse peak time
 #    )
	
	# model1 = (ω0 = 5e-3, Ω0 = model.Ω0, Nγ = model.Nγ, tγ = model.tγ)
	# model2 = (ω0 = 1e-1, Ω0 = model.Ω0, Nγ = model.Nγ, tγ = model.tγ)
	
	# τs = t0 .- reverse(t0)
	# Θτ = (τ -> τ > 0 ? 1 : τ == 0 ? 0.5 : 0.0).(τs)
	
	# xs = diag(reverse((data.TcG.data - data.TcL.data), dims = 2))
	
	# y1s = [EchoPulse.GγG0_ext(τ/2, -τ/2; model = model1) - EchoPulse.GγL0_ext(τ/2, -τ/2; model = model1) for τ in τs]
	# y2s = [EchoPulse.GγG0_ext(τ/2, -τ/2; model = model2) - EchoPulse.GγL0_ext(τ/2, -τ/2; model = model2) for τ in τs]

	# zs = EchoPulse.KadanoffBaym.ft(τs, xs .* ys; inverse = false)
	# zs = let
	# 	x1 = EchoPulse.KadanoffBaym.ft(τs, y2s; inverse = false)
	# 	x2 = EchoPulse.KadanoffBaym.ft(τs, xs; inverse = false)
	# 	EchoPulse.KadanoffBaym.ft(x1[1], x1[2] .* x2[2]; inverse = true)
	# 	# x1[1], x1[2] .* x2[2]
	# end
	# plot(zs[1], zs[2] |> real)

	# xlims = (-200, 200)
	# plot!([Inf], [Inf]; 
	# 	line = (1.5, :solid, :black), 
	# 	label=L"T^R_a",
	# 	subplot = 2)
	# plot!(τs, im * y2s |> real, 
	# 	line = (2.0, :dot, colours[end]), 
	# 	xlims = xlims,
	# 	label = nothing,
	# 	xlabel = L"\tau\ \upsilon", 
	# 	# yaxis = nothing,
	# 	# ylabel = L"-\operatorname{Im} G^R_a(\,\cdot, \tau)_{\mathrm{W}}",
	# 	yguidefontsize = 10,
	# 	subplot = 2
	# )
	# annotate!(345, 3ylims[end] / 4 - 1e-4, text(L"-\operatorname{Im} G^R_a(\,\cdot, \tau)_{\mathrm{W}}", 18, rotation = 90, "Computer Modern"),)
	
	# plot!(τs, im * y1s |> real, 
	# 	line = (2.0, :dot, colours[2]),
	# 	label = nothing,
	# 	subplot = 2)
	# plot!(τs, inv(model.Γ0) * im * xs |> real, 
	# 	line = (3.0, :solid, :black), 
	# 	label=nothing,
	# 	# ylabel = L"-\operatorname{Im} [i T^>_a - i T^<_a](\,\cdot, \tau)_{\mathrm{W}}",
	# 	xlims = xlims,
	# subplot = 2)
	# # plot!(framestyle = :box,)
	# # savefig("t-matrix.svg")

	# plot!(framestyle = :box,)

	# savefig("reflectivity-3.svg")
end

# ╔═╡ Cell order:
# ╟─9c40c777-cdd0-4436-ba63-858226cc050d
# ╠═1e759b6c-8b76-11ed-327f-456a27dc60d5
# ╠═40da4bf1-97b5-4b7a-a47c-495e425e40f5
# ╠═fdf9c03e-d8fc-4218-9c46-7d0b165811f5
# ╠═22c04e92-eaa8-4ca2-8388-9ba24db1672f
# ╠═20dcd8bd-042c-4340-9ddd-929e5e3b64f4
# ╠═ce2c9fca-1345-457e-b43b-5c11ff8f14db
# ╠═aef331b5-c5b3-4eb2-8713-278904861477
# ╠═8af64e39-b83c-48d0-aea9-cd90ddcec879
# ╠═6001df42-9cc0-483f-9578-e56e4c8ddf8f
# ╠═dc8dc52c-3900-4bfa-a1b4-12b5f487fe0b
