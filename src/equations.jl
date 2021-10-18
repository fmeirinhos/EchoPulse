function fv!(out, ts, t1, t2, model::PhotonAssistedSIAM, data::TimeData, T₀::Float64)
    @unpack GfL, GfG, GbL, GbG, ΣfL, ΣfG, ΣbL, ΣbG, GcL, GcG, TcL, TcG, ωc = data

    ∫dt(x...) = integrate(ts, t1, t2, x...)

    # pseudo fermion
    out[1] = -im * (∫dt(1, t1, ΣfG, GfL) - ∫dt(1, t2, ΣfL, GfG))
    out[2] = -im * (∫dt(t2, t1, ΣfG, GfG))

    # slave boson
    out[3] = -im * (-model.ε₀f * GbL[t1, t2] + ∫dt(1, t1, ΣbG, GbL) - ∫dt(1, t2, ΣbL, GbG))
    out[4] = -im * (-model.ε₀f * GbG[t1, t2] + ∫dt(t2, t1, ΣbG, GbG))

    # laser pulse
    # @unpack GγL, GγG, TγL, TγG = data
    # h0 = model.ω₀ - 1.0im * (ts[t1] - ts[t2]) * model.Ω₀^2
    # h1 = 0.5im * (T₀ - 0.5(ts[t1] + ts[t2])) * model.Ω₀^2

    # out[5] = begin
    #     # I = ∫dt(1, t1, TγG, GγL) - ∫dt(1, t2, TγL, GγG) - ∫dt(t2, t1, TγL, GγL)
    #     -im * ((h0 + h1) * GγL[t1, t2]) #+ (abs2(I) > 1e-5 ? I : 0.0))
    # end
    # out[6] = begin
    #     # I = ∫dt(t2, t1, TγG, GγG) - ∫dt(1, t1, TγL, GγG) + ∫dt(1, t2, TγG, GγL)
    #     -im * (h0 * GγG[t1, t2] + h1 * GγL[t1, t2]) #+ (abs2(I) > 1e-5 ? I : 0.0))
    # end

    # c-electrons
    # if !isnothing(ωc)
    #     for i in eachindex(ωc)
    #         out[6+i] = -1.0im * (ωc[i] * GcL[i][t1, t2] + ∫dt(1, t1, TcG, GcL[i]) - ∫dt(1, t2, TcL, GcG[i]) + ∫dt(t1, t2, TcL, GcL[i]))
    #         out[6+i+length(ωc)] = -1.0im * (ωc[i] * GcG[i][t1, t2] + ∫dt(t2, t1, TcG, GcG[i]) - ∫dt(1, t1, TcL, GcG[i]) + ∫dt(1, t2, TcG, GcL[i]))
    #     end
    # end

    return out
end

function fd!(out, ts, t1, t2, model::PhotonAssistedSIAM, data::TimeData, T₀)
    fv!(out, ts, t1, t2, model, data, T₀)
    out .-= adjoint.(out)
end

function Σ!(ts, t1, t2, model::PhotonAssistedSIAM, data::TimeData)
    @unpack GcL, GcG, GfL, GfG, GbL, GbG, GγL0, GγG0, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG, dos_c = data

    ρcL = isnothing(dos_c) ? GcL(ts[t1], ts[t2]) : sum(GcL[i][t1, t2] * dos_c[i] for i in eachindex(dos_c))
    ρcG = isnothing(dos_c) ? GcG(ts[t1], ts[t2]) : sum(GcG[i][t1, t2] * dos_c[i] for i in eachindex(dos_c))

    V²L = model.V₀^2 + im * model.Γ₀^2 * (GγL0(ts[t1], ts[t2]) + GγG0(ts[t2], ts[t1]))
    V²G = model.V₀^2 + im * model.Γ₀^2 * (GγG0(ts[t1], ts[t2]) + GγL0(ts[t2], ts[t1]))
    # V²L = model.V₀^2 + im * model.Γ₀^2 * (data.GγL[t1, t2] + data.GγG[t2, t1])
    # V²G = model.V₀^2 + im * model.Γ₀^2 * (data.GγG[t1, t2] + data.GγL[t2, t1])

    GdL = 1.0im * GfL[t1, t2] * GbG[t2, t1]
    GdG = 1.0im * GfG[t1, t2] * GbL[t2, t1]

    ΣfL[t1, t2] = +1im * V²L * GbL[t1, t2] * ρcL
    ΣfG[t1, t2] = +1im * V²G * GbG[t1, t2] * ρcG
    ΣbL[t1, t2] = -2im * V²L * GfL[t1, t2] * ~ρcG
    ΣbG[t1, t2] = -2im * V²G * GfG[t1, t2] * ~ρcL

    TcL[t1, t2] = +0.3im * model.V₀^2 * GfL[t1, t2] * GbG[t2, t1]
    TcG[t1, t2] = +0.3im * model.V₀^2 * GfG[t1, t2] * GbL[t2, t1]

    TγL[t1, t2] = +0.3im * model.Γ₀^2 * (GdL * ~GdG + GdL * ~GdG)#(GdL * ~ρcG + ρcL * ~GdG)
    TγG[t1, t2] = +0.3im * model.Γ₀^2 * (GdG * ~GdL + GdG * ~GdL)#(GdG * ~ρcL + ρcG * ~GdL)
    return
end

function resize_Σ!(data::TimeData)
    @unpack GfL, ΣfL, ΣfG, ΣbL, ΣbG, TcL, TcG, TγL, TγG = data
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
    u0 = [data.GfL, data.GfG, data.GbL, data.GbG]#, data.GγL, data.GγG]
    return isnothing(data.dos_c) ? u0 : [u0; data.GcL; data.GcG]
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
