# Euphorion — Parallel Biclique Search with Asymmetric GPU Compute

> *"Lasst mich im düstern Grab nicht allein!"*
> — Euphorion, Faust II Act 3

Born of Faust (modern) and Helena (classical) — Euphorion fuses two worlds.
Biclique's two sides map naturally to H100 (dense matrix ops) and A6000 (sparse traversal),
with GREAT's sampling for approximate acceleration.

## Upstream

| Directory | Origin | Role |
|-----------|--------|------|
| `upstream/mosib` | [nedchu/mosib-release](https://github.com/nedchu/mosib-release) | Most similar biclique search at scale |
| `upstream/great` | [sinhong-cheuk/GREAT](https://github.com/sinhong-cheuk/GREAT) | Reservoir sampling triangle counting on streams |
