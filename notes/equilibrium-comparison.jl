### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b9bc6a60-9622-11eb-3b18-9b0d614ab391
begin
	using Revise
	using DSP, NCA
	using NCA: nb, nf
	using NLsolve
	using QuadGK
	using Plots, LaTeXStrings
	Plots.plotly()
	#Plots.gr()
end;

# ╔═╡ 36bbe400-9623-11eb-31e8-3f197386e7cc
# parameters (units of half-bandwidth)
begin
	β  = 1.0e3  		# initial temperature

	model = PhotonAssistedSIAM(
		# siam
		V₀=sqrt(0.08),	# hybridization strength
		ε₀f=-0.25,		# |d-electron energy

		# quantum light
		Γ₀=sqrt(0.005), # light-matter coupling
		ω₀=2e-2, 		# photon energy
		Ω₀=2e-2,		# frequency bandwitdh of the pulse
	)
	
	# frequency mesh (increase the number of points if it fails)
	mesh = LinearMesh(-3.0, 3.0, 2^14)

	# custom bose distribution function
	f(x) = 1/(exp(-β * x) + 1)
	# f(x) = 1/(exp(10 * x) - 1)	
end;

# ╔═╡ 03ada4c6-9624-11eb-376f-bffd9c9362c3
begin
	# c-electron dos
	dos(ω) = 2π * NCA.gaussian(ω, 0.0, 1/3)
		
	# nca-problem
	nca_photon = let		
		# photon dos
		ργ = 2π * NCA.gaussian.(mesh, model.ω₀, model.Ω₀)

		FreqNCA(
			β=β,
			mesh=mesh, 
			model=model, 
			data=PhotonAssistedSIAMData(model, mesh, β, dos.(mesh), ργ, f.(mesh))
		)
	end
	
	# nca-problem
	nca_nophoton = let
		model_=SIAM(V₀=model.V₀, ε₀f=model.ε₀f)
		
		FreqNCA(
			β=β,
			mesh=mesh, 
			model=model_,
			data=SIAMData(model_, mesh, β, dos.(mesh))
			)
	end
end; 

# ╔═╡ 1a291798-96b6-11eb-3652-55f648271252
begin
	fixed_point(f, x0) = let
	  x0 = nlsolve(x -> f(x) - x, x0; 
		show_trace=:true,
		method=:anderson, 
		beta=0.15,# Since DMFT is kind of unstable we take very conservative steps
		m=1,      # and ignore previous function calls
		ftol=1e-1 # until we are close to the basin of convergence
		).zero

	  nlsolve(x -> f(x) - x, x0; 
		show_trace=:true,
		method=:anderson, 
		beta=0.4,
		ftol=1e-3
		).zero
	end;
	
	function solve!(nca)
	  # Initial guesses
	  x0 = cat(
			hcat(-0.1nca.data.ρc, -0.1nca.data.ρc),
			hcat(-0.1nca.data.ρc, +0.1nca.data.ρc),
			dims=3
			)

	  imΣ = fixed_point(x0) do imΣ′
		imG = NCA.slave_particle_imG(nca, imΣ′)
		imΣ = NCA.slave_particle_imΣ(nca, imG)
		imΣ - imΣ′
	  end

	  imΣ, NCA.projector(nca, NCA.slave_particle_imG(nca, imΣ))
	end;
	
	function solve_normal!(nca)
	  # Initial guesses
	  x0 = hcat(0.05nca.data.ρc, 0.05nca.data.ρc)

	  imΣ = fixed_point(x0) do imΣ′
		Σ = NCA.slave_particle_Σ!(nca, imΣ′)
		_ = NCA.calculate_λ!(nca, imΣ′, Σ)
		imG = NCA.slave_particle_imG!(nca, imΣ′, Σ)
		imΣ = NCA.slave_particle_imΣ!(nca, imG)

		imΣ - imΣ′
	  end

	  imΣ, NCA.projector(nca, NCA.slave_particle_imG!(nca, imΣ))
	end;

	
	# This is a rudimentary DMFT algorithm. It's only stable at high T and N<2^13
	dmft = false
	
# 	function dmft_solve!(nca)
# 	  x0 = hcat(nca.data.ρc, 0.05nca.data.ρc, 0.05nca.data.ρc)
# 	  imΣ = fixed_point(x0) do x
# 		# guess for the local self-energies and bath GF
# 		imΣ = x[:,2:3]
# 		Gc = NCA.kramerskronig(nca.mesh, x[:,1], :real) + 1.0im * x[:,1]

# 		# update the local c-electron DOS (needed for `nca`-dependent functions)
# 		nca.data.ρc = 2imag(Gc)
# 		if nca.model isa SIAM
# 			nca.data.ρc_th½ = 2imag(Gc .* nca.data.th½)
# 		end

# 		# NCA step: calculate local self-energies and Gd
# 		NCA.calculate_λ!(nca, imΣ)
# 		imG = NCA.slave_particle_imG!(nca, imΣ)
# 		imΣ′ = NCA.slave_particle_imΣ!(nca, imG)
# 		_, Gd = NCA.projector(nca, imG)    

# 		# local self-energy
# 		Σc = @. 1 / (inv(Gd) / nca.model.V₀^2 + Gc)

# 		# local propagator 
# 		Gloc = let
# 		  g = similar(Σc)

# 		  for (i, ωᵢ) in enumerate(nca.mesh)
# 			g[i] = quadgk(ε -> dos(tan(ε) * imag(Σc[i]) + ωᵢ - real(Σc[i])), -pi/2, pi/2)[1]/(2pi)
# 		  end

# 		  NCA.kramerskronig(nca.mesh, g, :real) + 1.0im * g
# 		end

# 		# new bath gf
# 		Gc′ = @. 1 / (inv(Gloc) + Σc)

# 		# tracking that the DOS is ok normalised
# 		@show NCA.integrate(nca.mesh, 2imag(Gc))/(2pi)
# 		@show NCA.integrate(nca.mesh, 2imag(Gc′))/(2pi)

# 		hcat(imag(Gc′), imΣ′)
# 	  end[:,2:3]

# 	  imΣ, NCA.projector(nca, NCA.slave_particle_imG!(nca, imΣ))
# 	end;
end;

# ╔═╡ 341da57e-9625-11eb-1ff3-2f0db2f8511a
begin
	imΣ_photon, (Gc_photon, Gd_photon) = solve!(nca_photon)
	imΣ_nophoton, (Gc_nophoton, Gd_nophoton) = solve_normal!(nca_nophoton)
end;

# ╔═╡ 5b95fa66-9625-11eb-3473-1bf9e12b7850
begin
	plot(nca_photon.mesh.mesh, imag(Gd_photon);
		label=L"A_d ~\textrm{(photon assisted)}",
		title=L"A_d ~\textrm{spectral function}", 
		xlab=L"\omega/D")

	plot!(nca_photon.mesh.mesh, imag(Gd_nophoton);
		label=L"A_d ~\textrm{(no photon)}")
	
	# plot!(nca_photon.mesh.mesh, 0.5nca_photon.data.ρc;
	# 	label=L"A^0_c",
	# 	line=(0.5,:grey))
	
	vline!([model.ε₀f];
		label=L"\varepsilon_d",
		line=(0.5,:dash,:green))
	
	vline!([-model.ω₀, model.ω₀];
		label=L"\omega_0",
		line=(0.5,:dash,:orange))

	# savefig("spectra-equilibrium.pdf")
end

# ╔═╡ 10323bf2-96b3-11eb-3030-f56891a769f0
begin
	plot(nca_photon.mesh.mesh, imag(Gc_photon);
		label=L"A_c ~\textrm{(photon assisted)}",
		title=L"A_c ~\textrm{spectral function}", 
		xlab=L"\omega/D")
	
	plot!(nca_photon.mesh.mesh, imag(Gc_nophoton);
		label=L"A_c ~\textrm{(no photon)}")
	
	vline!([model.ε₀f];
		label=L"\varepsilon_d",
		line=(0.5,:dash,:green))
	
	vline!([-model.ω₀, model.ω₀];
		label=L"\omega_0",
		line=(0.5,:dash,:orange))
end

# ╔═╡ 4457d152-9626-11eb-19f9-ebe48f80ed2a
begin
	plot(nca_photon.mesh.mesh .+ nca_nophoton.model.λ, -0.5 * imΣ_photon[:,1,1];
		label=L"\Sigma_b ~\textrm{(photon assisted)}",
		title=L"\Sigma_b ~\textrm{self-energy}", 
		xlab=L"\omega/D")
	
	plot!(nca_photon.mesh.mesh, nca_nophoton.data.fermi⁻ .* imΣ_nophoton[:,1,1];
		label=L"\Sigma_b ~\textrm{(no photon)}")
	
	vline!([model.ε₀f];
		label=L"\varepsilon_d",
		line=(0.5,:dash,:green))
	
	vline!([-model.ω₀, model.ω₀];
		label=L"\omega_0",
		line=(0.5,:dash,:orange))
end

# ╔═╡ af082812-9626-11eb-11b9-016989864cf5
begin
	plot(nca_photon.mesh.mesh .+ nca_nophoton.model.λ, -0.5 * imΣ_photon[:,2,1];
		label=L"\Sigma_f ~\textrm{(photon assisted)}",
		title=L"\Sigma_f ~\textrm{self-energy}", 
		xlab=L"\omega/D")
	
	plot!(nca_photon.mesh.mesh, nca_nophoton.data.fermi⁻ .* imΣ_nophoton[:,2,1];
		label=L"\Sigma_f ~\textrm{(no photon)}")
	
	vline!([model.ε₀f];
		label=L"\varepsilon_d",
		line=(0.5,:dash,:green))
	
	vline!([-model.ω₀, model.ω₀];
		label=L"\omega_0",
		line=(0.5,:dash,:orange))
end

# ╔═╡ Cell order:
# ╟─b9bc6a60-9622-11eb-3b18-9b0d614ab391
# ╠═36bbe400-9623-11eb-31e8-3f197386e7cc
# ╟─03ada4c6-9624-11eb-376f-bffd9c9362c3
# ╟─1a291798-96b6-11eb-3652-55f648271252
# ╟─341da57e-9625-11eb-1ff3-2f0db2f8511a
# ╟─5b95fa66-9625-11eb-3473-1bf9e12b7850
# ╟─10323bf2-96b3-11eb-3030-f56891a769f0
# ╟─4457d152-9626-11eb-19f9-ebe48f80ed2a
# ╟─af082812-9626-11eb-11b9-016989864cf5
