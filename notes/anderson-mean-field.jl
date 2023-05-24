### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ 7a5d3c22-e316-11ec-2a93-c708bab8e197
begin
	import Pkg; Pkg.activate()
	using QuadGK
	using NLsolve
	import PyPlot
	using LaTeXStrings
end;

# ╔═╡ 3fe95e22-bc3f-40e8-8943-119385340147
begin
	nf(ω, β) = 0.5 * (1.0 - tanh(0.5β * ω))
end

# ╔═╡ f9c403a5-0421-4781-a1d3-ae62655e3733
begin
	function f(x, params)
		(; εd, U, β) = params
		res1 = 1/pi * quadgk(ω -> nf(ω,β) / (1 + (ω - (εd + U * x[2]))^2), -Inf, +Inf)[1]
		res2 = 1/pi * quadgk(ω -> nf(ω,β) / (1 + (ω - (εd + U * x[1]))^2), -Inf, +Inf)[1]
		[res1, res2] - x
	end
end

# ╔═╡ 0a14b2ba-c2ae-41cd-8498-7ca1f36cbcba
begin
	εs = range(-10.0, 0.0, length=300)
	Us = range(0.0, +10.0, length=300)

	sols = []
	for u in Us
		for ε in εs
			p = (εd = ε, U = u, β=1e4)
			sol = nlsolve(x -> f(x, p), [0.8, 0.0];
				show_trace=false, 
				method=:newton, 
				ftol=1e-8, 
				iterations=10)
			push!(sols, sol.f_converged ? sol.zero : zero(sol.zero))
		end
	end

	sols = reshape(sols, length(εs), length(Us))
end;

# ╔═╡ d7274904-d847-4d90-bc63-2e4adc6d3076
magnetization = map(x -> abs(x[1] - x[2]), sols) .|> x -> clamp(x, 1e-6, 1.0)

# ╔═╡ f043e915-5b49-45dd-a706-414ecf43f2f7
begin
  PyPlot.rc("text", usetex=true)
  PyPlot.rc("text.latex", preamble=["\\usepackage{lmodern}", "\\usepackage{amsmath}", "\\usepackage{amssymb}"])
  PyPlot.rc("pgf", rcfonts=false)

  # Set y axis
  PyPlot.rc("ytick", direction = "inout")
  PyPlot.rc("ytick.major", size = 3)
  PyPlot.rc("ytick.major", width = 0.5)
  PyPlot.rc("ytick.minor", size = 1.5)
  PyPlot.rc("ytick.minor", width = 0.5)
  PyPlot.rc("ytick.minor", visible = true)
  PyPlot.rc("ytick", right = true)

  # Set x axis
  PyPlot.rc("xtick", direction = "inout")
  PyPlot.rc("xtick.major", size = 3)
  PyPlot.rc("xtick.major", width = 0.5)
  PyPlot.rc("xtick.minor", size = 1.5)
  PyPlot.rc("xtick.minor", width = 0.5)
  PyPlot.rc("xtick.minor", visible = true)
  PyPlot.rc("xtick", top = true)

  # Always save as 'tight'
  PyPlot.rc("savefig", bbox = "tight")
  PyPlot.rc("savefig", pad_inches = 0.05)

  # Set line widths
  PyPlot.rc("axes", linewidth = 0.5)
  PyPlot.rc("grid", linewidth = 0.5)
  PyPlot.rc("lines", linewidth = 1.0)
    
  # Remove legend frame
  PyPlot.rc("legend", frameon = false)

  PyPlot.rc("axes.spines", left = true)
  PyPlot.rc("axes.spines", bottom = true)
  PyPlot.rc("axes.spines", top = true)
  PyPlot.rc("axes.spines", right = true)

  # For my thesis
  PyPlot.rc("figure", figsize = [4, 3])
  PyPlot.rc("axes", labelsize = 11)
  PyPlot.rc("font", size = 11)
  PyPlot.rc("legend", fontsize = 9)
  PyPlot.rc("xtick", labelsize = 9)
  PyPlot.rc("ytick", labelsize = 9)
end

# ╔═╡ 885e06da-b82c-4544-b1e0-ade36a1cf91a
begin
	PyPlot.clf()
	base = PyPlot.contourf(εs, Us, magnetization', levels=5, 
		cmap = "Spectral_r", alpha=0.8)
	line = PyPlot.contour(εs, Us, magnetization', levels = base.levels, colors="black", linewidths=1.0)
	# PyPlot.title("Local moment formation")
	PyPlot.xlabel(L"\varepsilon^f_0 / \Delta")
	PyPlot.ylabel(L"U / \Delta")
	PyPlot.clabel(line, inline=1, fontsize=8)
	bar = PyPlot.colorbar(base)
	bar.add_lines(line)
	bar.ax.set_ylabel("magnetization")
	PyPlot.savefig("mf.pdf")
	# PyPlot.show()
end

# ╔═╡ Cell order:
# ╠═7a5d3c22-e316-11ec-2a93-c708bab8e197
# ╠═3fe95e22-bc3f-40e8-8943-119385340147
# ╠═f9c403a5-0421-4781-a1d3-ae62655e3733
# ╠═0a14b2ba-c2ae-41cd-8498-7ca1f36cbcba
# ╠═d7274904-d847-4d90-bc63-2e4adc6d3076
# ╠═f043e915-5b49-45dd-a706-414ecf43f2f7
# ╠═885e06da-b82c-4544-b1e0-ade36a1cf91a
