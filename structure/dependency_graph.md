# Claim Dependency Graph

Generated from `\depends{}` annotations in `paper/sections/*.tex`.
Last synced: 2026-03-26.

```mermaid
graph TD
    c0_universal["c0_universal<br/>All probes same c0<br/>(§3)"]
    c0_is_omegamh2["c0_is_omegamh2<br/>c0 = Omega_m h^2<br/>(§3)"]
    c0_tensions_sign["c0_tensions_sign<br/>BAO +2.2sigma, SN mild<br/>(§4)"]
    bao_constrains_omh2["bao_constrains_omh2<br/>BAO sigma_c0 > 1<br/>(§4)"]
    w0wa_is_c0["w0wa_is_c0<br/>w0wa = c0 repackaged<br/>(§1, §5)"]
    c0_dominant_w0wa["c0_dominant_w0wa<br/>c0 largest grid range<br/>(§5)"]
    freed_calpha_pattern["freed_calpha_pattern<br/>c1 carries no info<br/>(§5)"]
    three_mode_ladder["three_mode_ladder<br/>w(0.46) = -1 +/- 0.05<br/>(§5)"]
    only_omk_measurable["only_omk_measurable<br/>Only Omk BAO new dir<br/>(§6)"]
    omk_coherence["omk_coherence<br/>c0/c1 -> same Omk<br/>(§6)"]
    sn_blind_curvature["sn_blind_curvature<br/>SN can't see curvature<br/>(§6)"]
    alens_dilutes["alens_dilutes<br/>A_lens reduces tension<br/>(§7)"]

    c0_universal --> c0_is_omegamh2
    c0_universal --> c0_tensions_sign
    c0_universal --> bao_constrains_omh2
    c0_universal --> only_omk_measurable
    c0_universal --> c0_dominant_w0wa

    c0_tensions_sign --> w0wa_is_c0
    c0_tensions_sign --> alens_dilutes

    w0wa_is_c0 --> freed_calpha_pattern
    w0wa_is_c0 --> three_mode_ladder

    only_omk_measurable --> omk_coherence
    only_omk_measurable --> sn_blind_curvature
```

## Reading the Graph

- **Root:** `c0_universal` -- the fundamental result that all probes share the same c0.
- **Arrows:** A -> B means claim A is a logical prerequisite for claim B.
- **Layer 1 (decomposition):** c0_universal -> c0_is_omegamh2, c0_tensions_sign, bao_constrains_omh2, only_omk_measurable, c0_dominant_w0wa
- **Layer 2 (reinterpretation):** c0_tensions_sign -> w0wa_is_c0, alens_dilutes
- **Layer 3 (w0wa details):** w0wa_is_c0 -> freed_calpha_pattern, three_mode_ladder
- **Layer 3 (curvature):** only_omk_measurable -> omk_coherence, sn_blind_curvature

## Claim count: 12 unique claims across 8 sections
