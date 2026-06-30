from mergen.sequential import extend, fill_around, subsample, run_order
import mergen, numpy as np, pandas as pd

space   = mergen.ParameterSpace({'x':[0,.2,.4,.6,.8,1.],'y':[0,.2,.4,.6,.8,1.]})
sampler = mergen.Sampler(space)

# 1. extend
ex = np.array([[0.2,0.4],[0.6,0.8]])
r = extend(sampler, ex, n_new=8, seed=44, verbose=False)
print(f"extend → {len(r.samples)} (=10)")

# 2. fill_around
r = fill_around(sampler, [[0.0,0.0],[1.0,1.0]], n_new=6, seed=44, verbose=False)
print(f"fill_around → {len(r.samples)} (=6)")

# 3. subsample
pool = np.random.default_rng(0).uniform(0,1,size=(50,2))
sel = subsample(sampler, pool, n_select=10)
print(f"subsample → {sel.shape}")

# 4. run_order
d = pd.DataFrame([[.1,.1],[.9,.9],[.5,.5]], columns=['x','y'])
o = run_order(sampler, d)
print(f"run_order → {o['run_order'].tolist()}")

# Sampler artık bu metodları taşımıyor
assert not hasattr(sampler, 'extend')
print("Sampler clean ✓")