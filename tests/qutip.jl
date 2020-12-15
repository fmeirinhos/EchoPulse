### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ e250a37a-3ec2-11eb-0805-ab8ce2558e19
begin
	using PyCall
	qt = pyimport("qutip")
	np = pyimport("numpy")
	
	ϵₖ = [0.0, 0.0]
	V₀ = sqrt(0.1)
	ε₀f = -0.3
	
	# number of fermions
	# (spin-degenerate)
	m = length(ϵₖ) + 2
	
	σᶻ = qt.destroy(2) * qt.destroy(2).dag() - qt.destroy(2).dag() * qt.destroy(2);
	
	operators = [[qt.qeye(2) for _ in 1:m+1] for _ in 1:m+1]
	cfb = [qt.qeye(2) for _ in 1:m+1]
	for k in 1:m
		operators[k][1:k] = [-σᶻ for _ in 1:k]
		operators[k][k] = qt.destroy(2)
	end

	# hard-core boson (Is this correct?)
	operators[m+1][m+1] = qt.destroy(2)

	for k in 1:m+1
		cfb[k] = qt.tensor(operators[k])
	end

	c = [cfb[k] for k in 1:length(ϵₖ)]
	f = [cfb[k] for k in length(ϵₖ)+1:m]
	b = cfb[m+1];

	# please verify
	H  = sum([V₀ * c[σ].dag() * b.dag() * f[σ] for σ in 1:length(ϵₖ)])
	H += H.dag()
	H += sum([0.0 * c[σ].dag() * c[σ] for σ in 1:length(ϵₖ)])
	H += sum([ε₀f * f[σ].dag() * f[σ] for σ in 1:length(ϵₖ)]);

	nc = 0.01
	nf = 0.3
	nb = 1 - 2nf

	ψ_c = [sqrt(1-nc) * qt.basis(2, 0) + sqrt(nc) * qt.basis(2,1) for _ in 1:length(ϵₖ)]
	# ψ_c[1] = qt.basis(2, 1) # c-electron is occupied
	ψ_f = [sqrt(1-nf) * qt.basis(2,0) + sqrt(nf) * qt.basis(2,1) for _ in length(ϵₖ)+1:m]
	ψ_b = [sqrt(1-nb) * qt.basis(2,0) + sqrt(nb) * qt.basis(2,1)] # boson is occupied

	ψ₀ = qt.tensor(vcat(ψ_c, ψ_f, ψ_b));

	# observables = vcat([c[k].dag() * c[k] for k in 1:length(ϵₖ)], [f[σ].dag() * f[σ] for σ in 1:m-length(ϵₖ)], [b.dag() * b]);
	observables = vcat([f[σ].dag() * f[σ] for σ in 1:m-length(ϵₖ)], [b.dag() * b]);

	dm = qt.Qobj(np.diag(np.diag(ψ₀ * ψ₀.dag())))
	
	times = 0:0.001:30
	# quickly solve once for observables
	me = qt.mesolve(H, dm, times, [], observables)
end

# ╔═╡ Cell order:
# ╠═e250a37a-3ec2-11eb-0805-ab8ce2558e19
