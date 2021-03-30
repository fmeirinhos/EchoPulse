module EchoPulse

using DSP, QuadGK, NCA, NLsolve
using DSP.FFTW, Interpolations, KadanoffBaym
using LinearAlgebra
using UnPack

import DSP.FFTW: fftfreq

include("utils.jl")
include("model.jl")
include("equations.jl")

export FrequencyData, TimeData

end # module