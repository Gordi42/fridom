# Active barotropic substeps and exact dependency radius

The periodic blocked kernel returns weighted averages only. Its instantaneous
final pressure and velocities are discarded. Trailing exactly zero filter
weights therefore need not advance the recurrence: their contributions vanish
for finite substep fields. Retain at least one step for an all-zero filter.
Internal and leading zero weights still advance the recurrence. The original
`dtau` remains unchanged; nominal substep count still sets the physical step.
The default 30-entry SM2005 filter has 21 active entries.

For horizontal x index i, one full forward-backward step computes pressure from
u[i-1], u[i] and local v. It then computes u from the new pressures p[i], p[i+1]
and local forcing. New v uses the new pressure at the same x index. Thus every
new component depends only on old components at x indices i-1 through i+1.
After B complete steps, width B is sufficient, including constant forcing
arrays. Pressure's leftward dependency and velocity's rightward dependency do
not add to a radius of two. The initial and final interior are separated from
the artificial periodic wrap of each extended local buffer by exactly B cells.

Select the largest divisor of the active step count no larger than local x
size. This guarantees each message is contained within one immediate neighbor
and executes no padding steps. A 32-cell local x domain now completes the
21-step default filter in one block with a 21-cell halo on each side. Arbitrary
positive filter lengths and smaller local domains retain multiple blocks.
Unsupported geometry and layouts continue through the existing field stencil.

Validation uses the complete untrimmed field-operator recurrence, random
pressure/velocity/forcing, widths equal to a shard, multiple blocks, and leading,
internal and trailing zero filter entries. Four-device tests explicitly pass
all device IDs and assert the actual device count. Reverse-mode kernel VJP
matches the complete recurrence; a short model-run quadratic-loss gradient
matches central differences. Independent random end-to-end centered/WENO5
oracles compare against the earlier one-device source, at unchanged tolerances.
