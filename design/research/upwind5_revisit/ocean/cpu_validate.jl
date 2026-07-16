using Oceananigans, Printf
for scheme in ("centered","upwind5","weno5")
    adv = scheme=="centered" ? Centered(order=2) :
          scheme=="upwind5" ? UpwindBiased(order=5) : WENO(order=5)
    n = 16
    grid = RectilinearGrid(CPU(); size=(n,n,n), x=(0,10000.0), y=(0,10000.0),
                           z=(0,100.0), topology=(Periodic,Periodic,Periodic))
    model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2,
        advection=adv, tracers=:b, buoyancy=BuoyancyTracer(), coriolis=FPlane(f=1e-4))
    kx=2π/10000.0; ky=2π/10000.0; kz=2π/100.0
    set!(model; u=(x,y,z)->0.2*sin(kx*x)*cos(ky*y), v=(x,y,z)->0.06*cos(kx*x),
         b=(x,y,z)->1e-4*cos(kz*z))
    for _ in 1:10; time_step!(model, 20.0); end
    u = Array(interior(model.velocities.u))
    @printf("%s: finite=%s maxu=%.4e\n", scheme, all(isfinite,u), maximum(abs,u))
end
