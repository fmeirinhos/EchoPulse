function fv!(out, ts, t1, t2, model::PhotonAssistedSIAM, data::TimeData, T₀::Float64; pulse)
    (; GfL, GfG, GbL, GbG, ΣfL, ΣfG, ΣbL, ΣbG, GcL, GcG, TcL, TcG, εs) = data

    ∫dt(x...) = integrate(ts, t1, t2, x...)

    # pseudo fermion
    out[1] = -im * (∫dt(1, t1, ΣfG, GfL) - ∫dt(1, t2, ΣfL, GfG))
    out[2] = -im * (∫dt(t2, t1, ΣfG, GfG))

    # slave boson
    out[3] = -im * (-model.ε₀f * GbL[t1, t2] + ∫dt(1, t1, ΣbG, GbL) - ∫dt(1, t2, ΣbL, GbG))
    out[4] = -im * (-model.ε₀f * GbG[t1, t2] + ∫dt(t2, t1, ΣbG, GbG))

    # laser pulse
    (; GγL, GγG, TγL, TγG) = data
    h0 = model.ω₀ - 1.0im * model.Ω₀ * 2tanh(model.Ω₀ * (ts[t1] - ts[t2]))
    h1 = -0.5im * model.Ω₀ * 2tanh(model.Ω₀ * (0.5(ts[t1] + ts[t2]) - T₀))

    out[5] = begin
        I = ∫dt(1, t1, TγG, GγL) - ∫dt(1, t2, TγL, GγG) - ∫dt(t2, t1, TγL, GγL)
        -im * ((h0 + h1) * GγL[t1, t2] + (pulse ? I : 0.0))
    end
    out[6] = begin
        I = ∫dt(t2, t1, TγG, GγG) - ∫dt(1, t1, TγL, GγG) + ∫dt(1, t2, TγG, GγL)
        -im * (h0 * GγG[t1, t2] + h1 * GγL[t1, t2] + (pulse ? I : 0.0))
    end

    # c-electrons
    if !isnothing(data.dos_c)
        for i in eachindex(εs)
            out[6 + i] = -im * (∫dt(1, t1, TcG, GcL[i], εs[i]) - ∫dt(1, t2, TcL, GcG[i], εs[i]) + ∫dt(t1, t2, TcL, GcL[i], εs[i]))
            out[6 + length(εs) + i] = -im * (∫dt(t2, t1, TcG, GcG[i], εs[i]) - ∫dt(1, t1, TcL, GcG[i], εs[i]) + ∫dt(1, t2, TcG, GcL[i], εs[i]))
        end
    end

    return out
end

function fd!(out, ts, t1, t2, model::PhotonAssistedSIAM, data::TimeData, T₀; pulse)
    fv!(out, ts, t1, t2, model, data, T₀, pulse=pulse)
    @. out -= conj(out)
end

function Σ!(ts, t1, t2, model::PhotonAssistedSIAM, data::TimeData)
    (; GcL, GcG, GfL, GfG, GbL, GbG, GγL0, GγG0, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, dos_c, εs) = data

    # NOTE: optimise this allocation
    ρcL = isnothing(dos_c) ? GcL(ts[t1], ts[t2]) : integrate(εs, [exp(-im * εs[i] * (ts[t1] - ts[t2])) * GcL[i][t1, t2] * dos_c[i] for i in eachindex(dos_c)])
    ρcG = isnothing(dos_c) ? GcG(ts[t1], ts[t2]) : integrate(εs, [exp(-im * εs[i] * (ts[t1] - ts[t2])) * GcG[i][t1, t2] * dos_c[i] for i in eachindex(dos_c)])

    # V²L = model.V₀^2 + im * model.Γ₀^2 * (GγL0(ts[t1], ts[t2]) + GγG0(ts[t2], ts[t1]))
    # V²G = model.V₀^2 + im * model.Γ₀^2 * (GγG0(ts[t1], ts[t2]) + GγL0(ts[t2], ts[t1]))
    V²L = model.V₀^2 + im * model.Γ₀^2 * (data.GγL[t1, t2] + data.GγG[t2, t1])
    V²G = model.V₀^2 + im * model.Γ₀^2 * (data.GγG[t1, t2] + data.GγL[t2, t1])

    ΣfL[t1, t2] = +1im * V²L * GbL[t1, t2] * ρcL
    ΣfG[t1, t2] = +1im * V²G * GbG[t1, t2] * ρcG
    ΣbL[t1, t2] = -2im * V²L * GfL[t1, t2] * ~ρcG
    ΣbG[t1, t2] = -2im * V²G * GfG[t1, t2] * ~ρcL

    GdL = im * GfL[t1, t2] * GbG[t2, t1]
    GdG = im * GfG[t1, t2] * GbL[t2, t1]

    TcL[t1, t2] = +model.γ * model.V₀^2 * GdL
    TcG[t1, t2] = +model.γ * model.V₀^2 * GdG

    TγL[t1, t2] = -model.γ * model.Γ₀^2 * im * (GdL * ~ρcG + ρcL * ~GdG) #(GdL * ~GdG + GdL * ~GdG)
    TγG[t1, t2] = -model.γ * model.Γ₀^2 * im * (GdG * ~ρcL + ρcG * ~GdL) #(GdG * ~GdL + GdG * ~GdL)
    return
end

function resize_Σ!(data::TimeData)
    (; GfL, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG) = data
    if (n = size(GfL, 2)) > size(ΣfL, 2)
        resize!(ΣfL, n)
        resize!(ΣfG, n)
        resize!(ΣbL, n)
        resize!(ΣbG, n)
        resize!(TcL, n)
        resize!(TcG, n)
        resize!(TγL, n)
        resize!(TγG, n)
    end
end

function initial_condition(data::TimeData)
    u0 = [data.GfL, data.GfG, data.GbL, data.GbG, data.GγL, data.GγG]
    return isnothing(data.dos_c) ? u0 : [u0; data.GcL; data.GcG]
end

function project(data::TimeData, model::PhotonAssistedSIAM; ts, dmft)
    # d-electron
    Gd = TimeOrderedGF(
        im * data.GfL.data .* transpose(data.GbG.data),
        im * data.GfG.data .* transpose(data.GbL.data),
        ts,
    )

    # free photon
    G0γ = TimeOrderedGF(
        data.GγL.data,
        data.GγG.data,
        ts)
    # G0γ = TimeOrderedGF(
    #     [data.GγL0(t1, t2) for t1 in ts, t2 in ts],
    #     [data.GγG0(t1, t2) for t1 in ts, t2 in ts],
    #     ts)

    # c-electron t-matrix
    Tc1 = model.V₀^2 * Gd
    Tc2 = im * model.Γ₀^2 * TimeOrderedGF(
        greater(Gd) .* (greater(G0γ) + transpose(lesser(G0γ))),
        lesser(Gd) .* (lesser(G0γ) + transpose(greater(G0γ))),
        ts)

    if !dmft
        # free c-electron
        G0c = TimeOrderedGF(
            [data.GcL(t1, t2) for t1 in ts, t2 in ts], 
            [data.GcG(t1, t2) for t1 in ts, t2 in ts], 
            ts)

        # WRONG: Correct formula can't use local G0c on the t-matrix part
        Ac = G0c + G0c ⋆ (Tc1 + Tc2) ⋆ G0c
    else
        # free c-electron
        G0c = [TimeOrderedGF(
          exp.(-im * ε * (ts .- ts')) .* data.GcL[i].data, 
          exp.(-im * ε * (ts .- ts')) .* data.GcG[i].data, 
          ts) for (i, ε) in enumerate(data.εs)]

        # c-electron spectral function
        Ac = EchoPulse.integrate(data.εs, (data.dos_c[i] * (G0c[i] + (G0c[i] ⋆ Tc1 ⋆ G0c[i])) for i in eachindex(data.εs)))
    end

    # photon t-matrix
    Tγ = -im * model.Γ₀^2 * TimeOrderedGF(
        lesser(Ac) .* transpose(greater(Gd)) + lesser(Gd) .* transpose(greater(Ac)),
        greater(Ac) .* transpose(lesser(Gd)) + greater(Gd) .* transpose(lesser(Ac)),
        ts,
    )

    # photon
    Gγ = G0γ + G0γ ⋆ Tγ ⋆ G0γ


    # Let's find Gc and Gγ
    #   Gγ = fixedpoint(G0γ) do x
    #       Tc2 = +1.0im * model.Γ₀^2 * TimeOrderedGF(
    #           greater(Gd) .* (greater(x) + transpose(lesser(x))),
    #           lesser(Gd) .* (lesser(x) + transpose(greater(x))),
    #           ts
    #           )

    #       Gc = G0c + G0c ⋆ (Tc1 + Tc2) ⋆ G0c

    #       Tγ = TimeOrderedGF(
    #           model.Γ₀^2 * (-1.0im) *
    #           (lesser(Gc) .* transpose(greater(Gd)) + 
    #               lesser(Gd) .* transpose(greater(Gc))),
    #           model.Γ₀^2 * (-1.0im) *
    #           (greater(Gc) .* transpose(lesser(Gd)) + 
    #               greater(Gd) .* transpose(lesser(Gc))),
    #           ts
    #       )

    #       Gγ = G0γ + G0γ ⋆ Tγ ⋆ G0γ
    #   end

    (Gd=Gd, Ac=Ac, G0γ=G0γ, Gγ=Gγ)
end

""" ~: A(τ) –> A(-τ) """
~(a) = -conj(a)

""" Calculates GR(ω) and G<(ω) in equilibrium """
function calculate_Gω(ωs, model, ΣR, ΣL)
    GR = zero(ΣR)
    @. GR[:, 1] = ωs + model.ε₀f - ΣR[:, 1]
    @. GR[:, 2] = ωs - ΣR[:, 2]
    @. GR = inv(GR)

    GL = GR .* ΣL .* conj(GR)
    
    # We normalise it by Q because the fixed-point of G<(ω) is scale invariant
    # f(α G<) = α G<,  ∀α
    GL *= 2π / integrate(ωs, imag(-GL[:,1] + 2GL[:,2]))
    
    return GR, GL
end
    
""" Calculates Σ>(t) and Σ<(t) in equilibrium """
function calculate_Σt(model, GcG, GcL, GγG, GγL, GG, GL)
    V2G = model.V₀^2 .+ im * model.Γ₀^2 * (GγG + ~GγL)
    V2L = model.V₀^2 .+ im * model.Γ₀^2 * (GγL + ~GγG)
    
    Σt = zeros(ComplexF64, length(GγG), 4)
    @. Σt[:,1] = -2im * V2G * GG[:,2] * ~GcL
    @. Σt[:,2] = +1im * V2G * GG[:,1] * GcG
    @. Σt[:,3] = -2im * V2L * GL[:,2] * ~GcG
    @. Σt[:,4] = +1im * V2L * GL[:,1] * GcL
    
    return Σt
end
