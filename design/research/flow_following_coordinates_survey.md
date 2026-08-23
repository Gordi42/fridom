---
status: frozen
date: 2026-08-23
---

# Flow-following vertical coordinates — z*, MOM6's ALE (isopycnal / hybrid), and the continuous-ALE implementations

Research report (see [`README.md`](README.md) for status). Question
(owner, 2026-08-23): how do z* and MOM6's isopycnal-following
coordinates work, what do they do where interfaces cannot follow the
isopycnals (mixed layers, deep convection), and is there a short path
to them on FRIDOM's moving-geometry design? Method: source reading of
MOM6 `dev/gfdl` (`src/ALE/*.F90`, the OM4_025 configuration), MITgcm
(`calc_r_star.F`, `integr_continuity.F`, `integrate_for_w.F`,
`mom_calc_rtrans.F`, the NLFS docs), NEMO 4.2.2 (`domvvl.F90`,
`domqco.F90`), MPAS-Ocean (`mpas_ocn_thick_ale.F`,
`mpas_ocn_diagnostics.F`, the ALE design document), and the papers
(Adcroft & Campin 2004, Campin et al. 2004, Shchepetkin & McWilliams
2005, Megann et al. 2022, Griffies, Adcroft & Hallberg 2020; Leclair &
Madec 2011 and Petersen et al. 2015 only through secondary sources).
AI-assisted (two research agents); the FRIDOM consequences are in
[`../plans/active/flow_following_coordinates_plan.md`](../plans/active/flow_following_coordinates_plan.md).
Copies of every fetched source sit in the session scratchpad only.

## Verdict

- **Two families, two mechanisms.** (1) *Analytic* coordinates whose
  map depends on a 2-D state variable — z* (`z = η + (1 + η/H) z*`),
  σ with a free surface. The mesh motion is `∂_t z|_{z*} = (1 +
  z*/H) ∂_t η`; every model realizes it through the **discrete
  free-surface update**, never an analytic `∂_t η`. (2)
  *State-built* target grids — isopycnal, hybrid (HYCOM1, HYBGEN),
  adaptive. MOM6 builds the target per column as an algebraic
  function of the current state once per `DT_THERM` and **remaps**
  onto it (vertical Lagrangian-remap, no mesh velocity anywhere);
  MPAS-Ocean and NEMO-z̃ instead *choose a dia-surface transport* that
  relaxes the layer thicknesses toward a target — continuous ALE.
- **The geometric conservation law is the whole game.** MITgcm's
  `rStarDhCDt`, NEMO's `ssh(Kaa)−ssh(Kbb)` and MPAS's `projectedSSH −
  oldSSH` are all the column sum of the same thickness-weighted
  transport divergence the tracer step uses; the relative vertical
  velocity is the residual and closes to zero at the surface. Campin
  et al. 2004: "providing [continuity and the tracer equation] are
  discretized in the same way, both in time and space, a uniform
  concentration will remain constant … and the model will conserve
  any tracer locally"; an independently computed `∂_t M` breaks
  constancy. Consequence for a continuous-ALE design: the geometry
  parameter must be **advanced by the same tendency the ALE term
  uses**.
- **Where isopycnals cannot be followed, MOM6 does not make the
  solve smooth — it clips.** HYCOM1 (OM4's coordinate): monotonize
  the density profile bottom-up, locate the target densities
  (out-of-range targets go to the surface/bottom, targets inside a
  density jump to the jump), then per interface
  `z_K = min(max(z_iso(K), z*_nominal(K)), bottom, z_max(K), z_{K−1} +
  h_max(K−1))` — the `max` with a nominal z* floor is what makes the
  mixed layer and a convecting column z-like; `MIN_THICKNESS` (1 mm)
  inflation keeps the layer count fixed; the remap tolerates 0/1 mm
  layers. The pure RHO mode instead *sorts the state* (convective
  adjustment at p = 0) and collapses every unmatched layer to 1 mm at
  the surface or bottom. ADAPTIVE (Hofmeister) diffuses interface
  positions toward isopycnals with a stratification-dependent
  diffusivity that vanishes in unstable water, with the same optional
  z* floor.
- **Continuous-ALE targets need volume-conserving thickness
  safeguards** (MPAS min/max sweep; NEMO `rn_zdef_max` clip with
  column-sum redistribution) and the explicit vertical advection of
  the relative velocity becomes the time-step limiter when interfaces
  move fast (Megann et al. 2022 moved NEMO-z̃ to mixed
  explicit/implicit vertical advection).

## 1. z* and the rescaled equations (Adcroft & Campin 2004)

```
z* = H (z − η)/(H + η),  z = η + (1 + η/H) z*,  z* ∈ [−H, 0]
thickness factor   z_r = (H + η)/H           (h_k = h_k^0 (1 + η/H))
mesh velocity      ∂_t z|_{z*} = (1 + z*/H) ∂_t η
relative velocity  w* = H/(H+η) [ w − (1 + z*/H) D_t η + (z* η/H²) v_h·∇H ]
continuity         ∂_t[(H+η)/H] + ∇_{z*}·[((H+η)/H) v_h] + ∂_{z*}[((H+η)/H) w*] = 0
free surface       ∂_t η + ∇·∫_{−H}^{0} ((H+η)/H) v_h dz* = P − E
hydrostatics       ∂_{z*} p/ρ0 + ((H+η)/H) g ρ/ρ0 = 0
pressure gradient  ∇_{z*} p'/ρ0 + (g ρ'/ρ0) ∇_{z*}[η (1 + z*/H)]   (the slope term, O(∇η/H))
```

"Motion associated with the pure barotropic mode is identically
horizontal with w* = 0" — the external-mode heave is absorbed by the
mesh and leaves the vertical CFL. The computational domain is fixed,
so no vanishing layers arise; the only singularity is `η → −H`
(MITgcm guards `rStarFac = 1 + η/H` with `hFacInf = 0.2`,
`hFacSup = 2.0`).

**MITgcm r\*.** `CALC_R_STAR`: `rStarFacC = (etaH + Ro_surf −
R_low)/R_col`, faces area-averaged, `rStarDhCDt = (rStarFacC −
rStarFacNm1C)/deltaTFreeSurf`, `rStarExpC = h^{n+1}/h^n`.
`INTEGR_CONTINUITY` (exactConserv): the thickness-weighted transport
with the *current* `hFac` is summed, `dEtaHdt = −hDivFlow/rA − EmP`,
and `INTEGRATE_FOR_W` builds `w*_{k−1/2} = w*_{k+1/2} − ∇·(h_k u) −
(∂_t η/H) h_k^0` from the bottom — zero at the surface to round-off
because `dEtaHdt` is *defined* as that column sum. Tracers step in
flux form with the old thickness and the Adams–Bashforth tendency
rescaled by `dh^n/dh^{n+1}` (`FREESURF_RESCALE_G`); momentum gets
the same mesh transport in `mom_calc_rtrans.F`. Step order: dynamics
on `dh^{n−1}` → `UPDATE_R_STAR` → pressure solve → continuity →
`CALC_R_STAR` → thermodynamics.

**NEMO z\*** (`domvvl`/`domqco`): `e3t = e3t_0 (1 + ssh/ht_0)`; the SSH
increment is distributed proportionally to the current thickness.
**ROMS** (S&M 2005 eq. 1.10, 1.19): `Δz_k = Δz_k^0 (1 + ζ/h)`, and the
vertical velocity is computed from the discrete continuity
`W_{k+1/2} = Σ_{k'≤k}[(ΔV^{n+1} − ΔV^n)/Δt + ∇·U]` rather than the
volume ("the latter is entirely controlled by change of ζ");
two-way temporal averaging of the barotropic fluxes makes the 3-D
discrete continuity hold exactly between baroclinic steps.

## 2. MOM6: vertical Lagrangian-remap

**Step structure** (`MOM.F90`, `MOM_ALE.F90`, `MOM_regridding.F90`).
Dynamics every `DT`, thermodynamics every `DT_THERM`, and right after
it `ALE_regridding_and_remapping` (OM4: `DT = 900`, `DT_THERM =
7200` — eight Lagrangian steps per remap). Between remaps the layers
are Lagrangian (`ṙ = 0`: thickness evolves by horizontal convergence
only). Griffies et al. 2020, Table 1:

```
h†           = h^n − Δt ∇_r·(h u)            horizontal advection (Lagrangian)
[hC]†        = [hC]^n − Δt ∇_r·(h C u)
h^{n+1}      = h^target                      regrid
δ_r w^(ṙ)    = −(h^target − h†)/Δt            diagnosed dia-surface transport
[hC]^{n+1}   = [hC]† − Δt δ_r(w^(ṙ) C†)        remap
```

The regrid is a pure column operation on `(h, T, S)` plus bathymetry
(ADAPTIVE also reads the four neighbours); no grid velocity exists;
the regrid never changes the column's total thickness (top and bottom
interfaces are fixed); the remap is of the extensive `u h` through an
exact sub-cell decomposition (`intersect_src_tgt_grids`,
`remap_src_to_sub_grid` with Hallberg's "adjust the thickest
sub-cell by the residual" trick, `remap_sub_to_tgt_grid`), PLM / PPM
/ PQM reconstructions (OM4: `PPM_H4`), velocities remapped as cell
means on the averaged neighbouring thicknesses. `nk` is fixed for the
run; vanished layers are carried at 0 or `MIN_THICKNESS` and the
remap tolerates them (a vanished target cell inherits the first
sub-cell's value).

**Coordinate modes** (`regrid_consts.F90`): `LAYER`, `ZSTAR`, `RHO`,
`SIGMA`, `HYCOM1`, `SIGMA_SHELF_ZSTAR`, `ADAPTIVE`, `HYBGEN`
(`SLIGHT` removed 2023).

*ZSTAR* (`build_zstar_column`): nominal spacing `dz_k` stretched by
`Σh/(depth + z0_top)`, integrated down from `η`; then a bottom-up
sweep `zInterface(k) = max(zInterface(k), zInterface(k+1) +
min_thickness)` stacks the layers below the seafloor at
`min_thickness` (the partial-step realization). State-independent.

*RHO* (`build_rho_column`): `convective_adjustment` first sorts `T,
S` **and `h`** by potential density at p = 0 ("seems questionable,
although it does avoid ambiguous sorting"); vanished source layers
are dropped (`copy_finite_thicknesses`); densities at `P_REF`
(2e7 Pa); `build_and_interpolate_grid` locates each target density
in the monotone piecewise polynomial — out-of-range targets return
the surface/bottom, targets inside a discontinuity return the jump
("results in vanished layers near the boundaries"), non-monotone
input is FATAL; `old_inflate_layers_1d` raises every layer to
`MIN_THICKNESS` and charges the thickest layer. **Mixed column**: the
whole slab lands in the one layer whose target interval brackets it;
every lighter-target layer is a 1 mm layer at the surface.

*HYCOM1* (`build_hycom1_column`, Bleck 2002; OM4's mode):

```fortran
    do k = nz-1, 1, -1            ! monotone, not single valued
      rho_col(k) = min( rho_col(k), rho_col(k+1) )
    enddo
    call build_and_interpolate_grid(... rho_col ... target_density ... z_col_new ...)
    nominal_z = 0. ; stretching = z_col(nz+1) / depth
    do k = 2, CS%nk+1              ! z* floor: the deeper of isopycnal and nominal
      nominal_z = nominal_z + (z_scale * CS%coordinateResolution(k-1)) * stretching
      z_col_new(k) = max( z_col_new(k), nominal_z )
      z_col_new(k) = min( z_col_new(k), z_col(nz+1) )
    enddo
    do k=2,CS%nk                   ! ceilings (top and bottom interfaces never move)
      z_col_new(K) = min(z_col_new(K), CS%max_interface_depths(K), &
                         z_col_new(K-1) + CS%max_layer_thickness(k-1))
    enddo
```

then `filtered_grid_motion` (the `REGRID_TIME_SCALE` lag, OM4: 0)
and `adjust_interface_motion` (bottom-up `MIN_THICKNESS`
enforcement). `REGRID_COMPRESSIBILITY_FRACTION` (OM4: 0.01) adds
artificial compressibility "solely to make homogeneous regions appear
stratified". OM4_025: 75 layers, `ALE_COORDINATE_CONFIG =
"HYBRID:hycom1_75_800m.nc,sigma2,FNC1:2,4000,4.5,.01"`, the first six
target densities lighter than any water (so the top layers are
always on their z* levels), `MAXIMUM_INT_DEPTH_CONFIG =
"FNC1:5,8000.0,1.0,.01"`, `MAX_LAYER_THICKNESS_CONFIG =
"FNC1:400,31000.0,0.1,.01"`, `MIN_THICKNESS = 0.001`,
`INTERPOLATION_SCHEME = P1M_H2`, `REMAPPING_SCHEME = PPM_H4`.
**Deep-convection column**: targets lighter than the mixed water →
`z_iso = 0` → pushed to their z* levels; targets inside the jump at
the mixed-layer base pile at that depth and are lifted by the
ceilings into a coarse z grid; leftovers sit at 1 mm. The state is
untouched.

*HYBGEN* (`MOM_hybgen_regrid.F90`, the HYCOM generator): per-column
minimum thicknesses `dp0k` (deep) / `ds0k` (shallow, σ-like between),
`fixlay` = always-fixed surface layers above `isotop`; interfaces
move *incrementally* toward the targets by entrainment fractions
limited by the Bleck–Benjamin cushion function, relaxed by
`HYBGEN_RELAX_PERIOD`; in the mixed layer the layers sit at their
minimum `dp0k` — fixed z levels.

*ADAPTIVE* (`coord_adapt.F90`): an isopycnal displacement from the
horizontal neutral-density curvature over the four neighbours,
`dh = Δ²σ · h / (∂σ/∂z)`, limited to half the upwind thickness; then
an implicit (tridiagonal) diffusion of interface positions in index
space with `K_grid ∝ ADAPT_TIME_RATIO · nz² · depth · [zoom/(Z_zoom +
z) + buoy · max(∂ρ/∂z, 0)/Δρ0 + background/depth]` — in unstable or
unstratified water the buoyancy term vanishes and the grid relaxes
toward a smooth near-uniform spacing; optional z* floor
(`ADAPT_DO_MIN_DEPTH`). The closest analogue to a mesh-velocity
design, and even it falls back to a z* floor.

**Initialization**: `REMAP_AFTER_INITIALIZATION` (default on) and
`REGRID_ACCELERATE_INIT` iterate regrid → remap of T, S from the
original grid so the initial grid is consistent with the state.

## 3. Continuous-ALE implementations

**MPAS-Ocean** (`ocn_vert_transport_velocity_top`,
`ocn_ALE_thickness`): the mesh is set by *choosing* the transport
through the layer tops,

```
h^ALE_k = h^rest_k + h^SSH_k + h^hf_k + h^min_k
h^SSH_k = ζ · W_k h^rest_k / Σ W h^rest         (z-level: W = (1,0,..); z*: W = 1; user weights)
w^t_k   = w^t_{k+1} − D_k − (h^ALE_k − h^n_k)/Δt,   w^t_1 = w^t_{kmax+1} = 0
```

with `projectedSSH = oldSSH − Δt Σ D_k` from the **same** fluxes, so
the recursion closes at the surface (the discrete GCL). The
thickness is then stepped "in the identical manner as the tracer
equation". `impermeable_interfaces` sets `w^t = 0` (idealized
isopycnal; "does not support massless layers"). z̃ adds two
prognostic fields, `∂_t D^lf = −(2π/τ)(D^lf − D')` and `∂_t h^hf =
−(D' − D^lf) − (2π/τ_hhf) h^hf + ∇·(κ ∇h^hf)`. Optional
`config_use_min_max_thickness`: a down-then-up sweep clamping `h` to
`[h_min, f_max h^rest]` with the displaced volume carried along
(column total unchanged).

**NEMO z̃** (`dom_vvl_sf_nxt`): `e3 = e3* + e3'`, `∂_t e3' = −D' + D^LF
+ ∇·(κ∇e3') − (2π/τ_z) e3'`, `∂_t D^LF = −(2π/τ̃)(D^LF − D')`, with
`rn_rst_e3t = 30` d, `rn_lf_cutoff = 5` d, a hard deformation clip
`|e3'| ≤ rn_zdef_max · e3_0 = 0.9 e3_0` (ctl_stop beyond), and the
column-sum error spread proportionally to `e3t`. Megann et al. 2022
needed restoration of `e3*`, FCT positivity, bottom regridding with a
1 m minimum, interface smoothing, Higdon's barotropic-consistency
correction and mixed explicit/implicit vertical advection to make it
stable with real topography (+15–18 % cost; z̃ off below 100 m).

## 4. Comparison (Griffies, Adcroft & Hallberg 2020 §5)

| | quasi-Eulerian (z*: MITgcm, NEMO, ROMS) | vertical ALE (NEMO z̃, MPAS) | Lagrangian-remap (MOM6, HYCOM) |
|---|---|---|---|
| grid fixed by | analytic coordinate, `∂_t h` from the discrete `∂_t η` | prescribed target, `w_grid = −(h^target − h^n)/Δt` | `h^{n+1} = h^target` after a Lagrangian step |
| dia-surface transport | residual of continuity | residual of continuity with the chosen target | the remap velocity |
| time step | vertical CFL on `w*` (barotropic heave removed) | same, smaller in practice; fast interface motion needs implicit vertical advection | no vertical CFL (remap spans cells) |
| GCL | thickness and η by the same divergence | same + barotropic consistency corrections | trivial (Lagrangian step) + conservative remap |
| unstratified columns | not an issue (target depends on η only) | explicit thickness safeguards | massless layers handled natively |
| spurious mixing | explicit vertical advection of fast motion | reduced for waves, not for balanced flow | the remap itself; high-order PPM/PQM needed |

## 5. Consequences for a `z = M(b, params)` + mesh-velocity design

1. Never evaluate `∂_t M` independently of the discrete parameter
   update; derive the mesh velocity from the parameter tendency the
   stepper actually applies, or make the parameter a prognostic with
   that tendency — then `J^{n+1} − J^n` and the ALE flux agree and
   constancy/conservation hold.
2. Thickness-weighted flux form with `J = ∂M/∂b`; multistep schemes
   need the `J^n/J^{n+1}` rescale or must step `Jθ` — or, as in a
   design whose ALE term is a separate tendency on an intensive
   field, the equivalence `∂_t(Jθ) = J∂_tθ + θ ∂_b ż` holds exactly
   when `∂_t J = D_b(ż)` discretely.
3. Pressure gradient: the slope term `(g ρ'/ρ0) ∇_b M` — `O(∇η/H)`
   for z*; steeper maps want S&M-2003-type schemes.
4. Safeguards only where `params` can drive `J → 0`: a lower bound on
   `1 + η/H` for z*; a volume-conserving min/max thickness sweep for
   isopycnal-type targets, plus the z* floor and ceilings of HYCOM1
   for the unstratified case.
5. Momentum takes the same mesh transport; face-centred `J` must be
   averaged consistently with the kinetic-energy / vertical-advection
   discretization.
