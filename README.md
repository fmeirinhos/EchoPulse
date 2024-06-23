# Replication Data of the dissertation "Kondo Collapse and Revival by Pulsed Light"

### Data & file overview

1. File List:
 - mathematica/langreths.nb
 - mathematica/pulse.nb
 - mathematica/rate.nb
 - mathematica/wigner-transform.nb
 - notes/anderson-mean-field.jl
 - notes/conduction.jl
 - notes/correlations.jl
 - notes/initial-pulse.jl
 - notes/mean-field.jl
 - notes/multi-plotter.jl
 - notes/plotter.jl
 - src/EchoPulse.jl
 - src/eom.jl
 - src/model.jl
 - src/nca.jl
 - src/solve.jl
 - src/utils.jl
 - LICENSE
 - Project.toml

### Sharing / access information

1. Licenses
   [MIT License](LICENSE)

2. Links to publications that cite or use the data:
> Costa Meirinhos, Francisco: Kondo Collapse and Revival by Pulsed Light – Bonn, 2024 – Dissertation, Rheinische Friedrich-Wilhelms-Universität Bonn, https://nbn-resolving.org/urn:nbn:de:hbz:5-76501
> Meirinhos, Francisco, 2024, "Replication Data for: "Kondo Collapse and Revival by Pulsed Light"", bonndata, https://doi.org/10.60507/FK2/C0YZ1B


### Methodological information

- The files under `scr/` define the mathematical procedures required to solve an external Gaussian pulse of light interacting with a Bethe Anderson lattice via the non-crossing approximation, the main motif of the dissertation
- The files under `notes/` generate several plots which are helpful in understanding the problem just exposed. Regarding what is in the dissertation, `mean-field.jl` generates the non-equilibrium saddle-point plots of the Chapter 5 of the thesis. Regarding the plots of Chapter 6, `correlations.jl` generates the data, `plotter.jl` the plots which constitute a single model run and `multi-plotter.jl` the plots which constitute several model runs.
The software required to reproduce these is in `Project.toml`.

- The files under `mathematica/` are scripts that can be used to understand certain aspects of the dissertation. These require the software Mathematica 10.0 or later.
