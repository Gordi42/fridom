"""Probe jax introspection APIs (memory_analysis, cost_analysis, HLO)."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("backend", jax.default_backend(), "count", jax.device_count())

x = jnp.ones((256, 256, 256), dtype=jnp.float64)


def f(a):
    return a[:-1] * 1.0 + a[1:] * 1.1


jit = jax.jit(f)
lowered = jit.lower(x)
compiled = lowered.compile()
y = compiled(x)
y.block_until_ready()

mem = compiled.memory_analysis()
print("=== memory_analysis type ===", type(mem))
for attr in dir(mem):
    if not attr.startswith("_"):
        try:
            print("  ", attr, "=", getattr(mem, attr))
        except Exception as e:  # noqa: BLE001
            print("  ", attr, "ERR", e)

cost = compiled.cost_analysis()
print("=== cost_analysis type ===", type(cost))
if isinstance(cost, dict):
    for k, v in cost.items():
        print("  ", k, "=", v)
else:
    print(cost)

txt = compiled.as_text()
print("=== HLO length chars ===", len(txt))
print("=== first 1200 chars ===")
print(txt[:1200])

# jaxpr line count
jaxpr = jax.make_jaxpr(f)(x)
print("=== jaxpr lines ===", len(str(jaxpr).splitlines()))

print("=== opt_barrier exists ===", hasattr(jax.lax, "optimization_barrier"))
