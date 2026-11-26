# SG-0 Stand-Still Gate — System Overview

A physics-honest digital twin of a stationary stargate: a warp ring held at
`v_s = 0` anchoring one mouth of a Morris–Thorne wormhole, with a live
operational model (fuel, wear, manifests, service), a gate network of deployed
mouths keeping real ephemeris schedules, a full RF link layer (dishes, rain,
weather, mesh relay), civil works (foundation, seismic, thermal expansion),
chronology bookkeeping, and a fabrication manual that stops honestly where the
physics runs out.

One file — `stargate.py` (~13,400 lines) — contains the physics, the runtime
model, the reports, and three front-ends: a pygame 3-D viewer (default), a
tkinter 2-D fallback (`--window2d`), and a text console (`--console`).

## The honest verdict

The machine is two different objects with two different verdicts:

- **The housing is real engineering.** A 557 t ring of Inconel 718, 27 REBCO
  coils at 19 kA × 9,600 turns held at 18 K, a 4.5 MWth D-He3 reactor
  (×0.62 direct conversion → 2.79 MWe), a 236 MJ SMES buffer riding through
  1.9 min of transient, 6 radiator sails at 985 K against a 1900 K limit,
  a 34-m X-band pad dish and 5-m mouth dishes — all aerospace-manufacturable
  to stated tolerances, all 15 mass lines in the BOM, every one spec'd with
  material + purpose + ops row + buildability rating.
- **The throat is open physics.** Holding `b0 = 3.35 m` open needs
  `rho+tau = -8.6e41 J/m3` of exotic stress-energy (`-7.5e26 kg` equivalent).
  Every candidate source in the ledger fails or is unpriced: quantum-field
  sourcing dies under the Ford–Roman bound 26 orders early; the real Casimir
  lattice holds `-69.9 kJ`, ~39-40 orders short; a Visser cut-and-paste shell
  is 40 orders short; MMP monopole black holes would work but none has ever
  been observed; a classical condensate is unpriced open physics.

The UI never blurs that line: build step 14 (throat formation) is marked
`NO FAB PATH` and cannot be signed off.

## What the model computes

### Spacetime
- Morris–Thorne throat: `b(r) = b0²/r`, zero redshift, flare-out `b'(b0) = -1`,
  embedding `z(r) = 2 b0 √(r/b0 − 1)`, proper length 13 m, crossing 1.34 s at
  the 10 m/s traverse speed.
- NEC/ANEC ledgers (`rho`, `tau`, `rho+tau`, geodesic integral), Ford–Roman
  quantum-inequality caps, strobe-pulse chain analysis (7.8e25 pulses at a
  1.7e-22 duty — fails), tidal shear at b0/2b0/4b0, person-safe throat radius.
- Chronology: per-mouth clock rates from `phi_c2` + `beta`, antisymmetric
  offsets (A sees B's offset = −B sees A's, verified to 1e-9), CTC existence
  `|offset| ≥ D/c`, per-link CTC onset under accumulated drift, teleport dwell
  solves, and the Hawking CPC flag on time-machine transits.
- Transit physics the fiction skips: a payload keeps its departure-mouth
  inertial velocity — pad→LUNA inherits **678 m/s**, pad→MARS **4.69 km/s**,
  pad→LEO **7.38 km/s** — plus the well-depth energy ledger
  (`gate_transit_dv` / `gate_transit_energy`), both logged on every manifest.
- Mass-flux backreaction: each crossing moves mass between mouths; the pair
  tracks net flux vs the 557 t instability budget.

### The sky (real ephemeris, not a backdrop)
- First-order eccentric Kepler for every mouth: Mars `e = 0.0934` (perihelic
  oppositions down to 0.37 AU), Moon `e = 0.0549`, Earth `e = 0.0167`.
- Lunar nodal regression — the Moon's orbit plane turns a full cycle per
  18.61-yr draconic period; eclipse geometry rides on it.
- Mouth position/velocity vectors, pad-frame elevation **and** azimuth
  (`mouth_azimuth_deg`), retarded light-time solves, aberration ~20″,
  Doppler tracking windows, solar-conjunction blackout scanning
  (`dial_window` returns `blackout_s` + `dialable_now`, and `recv_dial`
  refuses a handshake through a dead carrier).
- Pass windows with real tracking specs: el/az slew rates, the keyhole flag
  (az rate > 1°/s beats a real dish — a physical criterion, not a heuristic),
  LEO ~13 passes/2 d, Mars/LUNA ~12-h star-like arcs once a day.
- LEO environment: beta-angle seasons (eclipse fraction 0.00–0.39 like the
  real ISS), ~5,691 charge cycles/yr on the eclipse book, ~2.24 km/month
  drag decay, array-skin thermal swing +124 °C → 3 K.
- J2000 civil epoch mapping (`pad_clock_epoch`), TT/TAI/GPS timescales
  (`pad_timescales`).

### The link (a real DSN worksheet, not a typed constant)
- `antenna_gt_dbk` — G/T derived from the billed apertures:
  `G/T = η(πDf/c)² − 10 log T_sys`; the 34-m pad lands 53.9 dB/K, the 5-m
  mouth 35.5 — not the ~76 dB/K a hand-waved constant claims.
- `link_budget_db` — FSPL, Earth's troposphere + f⁻² ionosphere on **every**
  above-horizon path (Mars rises through the same air as a star), DERIVED
  pointing loss `12(θ_err/HPBW)² = 0.227 dB` off the real beams
  (`antenna_beamwidth_deg`: pad 0.0735°, mouth 0.50°), rain fade
  (`link_weather_db`, ITU-R P.838 k·R^α·L — 0.63 dB at 15 mm/h), Shannon
  ceiling on a 10 MHz channel. Mars opposition: **937 kbps, ~5 GB/sol** —
  real Mars-DTE class.
- `pad_weather_ledger` — rain + wind-stow availability **99.3%**; every
  downlink book now reports scheduled GB *and* weather-derated `gb_expected`.
- `gate_mesh_downlink_gb` — mouth-to-mouth relay when the pad can't see the
  far mouth: capacity = the thinnest hop (MARS via LEO-RELAY: the 10 kbps
  hop binds against the 111 GB pad leg — relays buy coverage, not aperture).
- `mouth_stationkeeping_ledger` — every orbiting mouth pays its real year:
  LEO 15.5 m/s (drag), LUNA 0 (the Moon drives), MARS 2.0 (MRO-class trim),
  HELIOTAIL 3.5 (SOHO/JWST-class libration keeping).

### The pad (a civil construction, not a set piece)
- `pad_foundation_ledger` — 557 t on the 32 m² plinth = **0.17 MPa** bearing;
  competent soil (0.3 MPa) is the binding constraint at 1.76× margin —
  the exotic machine is held up by ordinary concrete.
- `pad_seismic_ledger` — 0.3 g design-basis: 1.64 MN lateral; overturning is
  the binding case at 1.6× margin (slide 2.0×).
- `pad_ring_thermal_mm` — the 32-m ring breathes **3.0 mm** over the diurnal
  swing: alignment is a thermal engineering problem, not just geometry.

### Machinery & operations
- `Gate`: chevron sector locks (9 sectors × 40°), iris firewall (matter
  admitted only when open), fuel, wear (0.1%/h → service ~41 d, collapse
  ~42 d), manifests, logs, persistence.
- `GatePair`: shared throat, `link_status`, `ping` (2.7 s throat RTT vs
  4.4 min light-time ≈ 98×), `hold_live`, `balance`, `transit`.
- `GateNetwork`: 9 registered sites, 6 deployed mouth-pairs, BFS routes,
  deploy-mission costs for mouth-less sites (Proxima is a 4.97 yr haul at
  β = 0.855 — distance is NOT irrelevant until a mouth exists).
- Addressing: 38-glyph alphabet, chevron sequences, `resolve_dial` lookup.
- `gate_parts_book()` — every billed mass line spec'd: material, purpose,
  derived mass, ops row, buildability. `gate_dims_glossary()` — all 89
  DIMS + PLANT keys carry units + purpose; an unglossed parameter fails
  the selftest.
- `site_hardware_spec(site)` — each remote site as a machine: derived
  antenna G/T, eclipse/daylight-derated power MW, eclipse-cycle battery
  book, hold-hardware mass, purpose.

## The four screens

**MODEL [M]** — the gate site rendered as ~30 hardware systems (torus shell,
collars, brace cage, cryostat shroud, 27 coil packs, 9 chevron wedges, 12-blade
iris, horizon ring, pedestal/pad, reactor+SMES tori, fuel tank, comms dish,
temperature-tinted radiators, mouth B on its wormhole arc). Data plate, ops
console, gauges, click-to-inspect spec pages.

**WORMHOLE [W]** — the physics page: animated two-sheet embedding, 16-row
throat ledger, payload-class verdicts, 90-day duty forecast, QI log-bars,
live transit verdicts, CTC status + onset, candidate-source table, QT
channel, teleport dwell line, 6-stage crossing pipeline, mass-flux ledger.

**NETWORK [N]** — deployed links with drift + CTC onset, log-radius mouth
map, site roster with registry notes, deploy costs, BFS routes, held-liveness.

**BUILD [B]** — the 14-step fabrication manual: procedure, sub-assembly
checklists, dependency gates (persisted to `build_state.json`), installed-mass
tracking, ~40-month critical path. Step 14 (throat) is honestly unmarkable.

## Math proofs — the pinned equations

Every number below is asserted by a named `--selftest` check; nothing is
typed for the story.

| # | Claim | Equation | Verified value | Enforcing check |
|---|-------|----------|----------------|-----------------|
| 1 | Throat shape | `b(r) = b0²/r`, `b'(b0) = −1` | flare-out holds | `Morris-Thorne throat satisfies b(r0)=r0` |
| 2 | Exotic bill | `E = −c⁴b0/6G` | −6.8e43 J ≈ −7.5e26 kg | `exotic bill ~2e43 J per metre` |
| 3 | QI hold | Ford–Roman τ_max(ρ) | 1.7e-26 s vs 1.34 s crossing | `QI hold << crossing for spec'd gate` |
| 4 | Clock rate | `dτ/dt = √(1+2Φ/c²)/γ` | LEO drift −24.5 µs/day | `site rate: LEO mean drift` |
| 5 | CTC onset | `\|offset\| ≥ D/c` | exact at boundary | `CTC boundary exact` |
| 6 | Transit Δv | payload keeps departure frame | 678 m/s LUNA, 4.69 km/s MARS | `transit dv` ×3 |
| 7 | Antenna G/T | `η(πDf/c)² − 10logT` | 53.9 dB/K @34 m X-band | `antenna: 34-m DSN-class` |
| 8 | Beamwidth | `HPBW ≈ 70λ/D` | 0.0735° | `beamwidth: 34-m pad` |
| 9 | Pointing loss | `12(θ_e/HPBW)²` | 0.227 dB | `pointing loss is DERIVED` |
| 10 | FSPL | `20log(4πd/λ)` | 266.4 dB @ opposition | `Mars FSPL ~266 dB` |
| 11 | Shannon ceiling | `B log2(1+SNR)` | 937 kbps @ Mars | `real Mars-DTE class` |
| 12 | Rain fade | `A = k R^α L_slant` | 0.63 dB @15 mm/h | `X-band rain fade` |
| 13 | Atmo delay | zenith ~10 m × 1/sin el, iono ∝ f⁻² | 3.7 m @45° Mars | `atmo delay` ×4 |
| 14 | Bearing | `P = mg/A` | 0.17 MPa, soil margin 1.76× | `foundation` ×2 |
| 15 | Seismic | `M_ot vs M_stab` | overturn 1.6× < slide 2.0× | `seismic` ×2 |
| 16 | Thermal growth | `ΔL = CTE·D·ΔT` | 3.0 mm | `ring thermal` ×2 |
| 17 | Drag decay | `dv/yr = v/(2a)·\|da/dt\|·yr` | 15.5 m/s/yr | `reboost/stationkeeping` |
| 18 | Eclipse | β-angle geometry | 0.00–0.39 seasonal | `eclipse profile` |
| 19 | Mesh relay | capacity = min(legs) | hop binds 0.11 GB | `mesh` ×4 |
| 20 | Lunar node | draconic 18.61 yr | flip ~9.3 yr | `lunar node` |

## Numbers that matter

| Quantity | Value | Meaning |
|---|---|---|
| throat b0 | 3.35 m | engineered mouth radius |
| rho+tau | −8.6e41 J/m³ | NEC violation at the wall |
| exotic bill | −6.8e43 J ≈ −7.5e26 kg | what holding it costs |
| QI hold | 1.72e-26 s | Ford–Roman bound (1 s pulse) |
| crossing | 1.34 s | needs 7.8e25× more hold |
| tidal at b0 | 1.6e15 g | lethal — safe b0 ≈ 1.35e8 m |
| installed mass | 557 t / gate | 15 billed lines incl. comms dish |
| pad G/T | 53.9 dB/K | derived, 34-m DSN class |
| Mars link | 937 kbps / ~5 GB/sol | opposition, weather-derated |
| mesh hop | ~10 kbps | mouth-to-mouth, 5-m dishes |
| bearing | 0.17 MPa | soil margin 1.76× is binding |
| seismic | overturn 1.6× | 0.3 g design basis |
| ring breathing | 3.0 mm/day | diurnal thermal growth |
| LEO keeping | 15.5 m/s/yr | drag decay bill |
| availability | 99.3% | rain + wind-stow derated |
| exotic source | none priced | Casimir −69.9 kJ, 39 orders short |
| mouth carry | Proxima 4.97 yr @ β 0.855 | distance is NOT irrelevant first |
| human teleport | 22.2 Gyr of channel | signals only, honestly |
| duty life | collapse ~41.7 d unserviced | wear kills before fuel does |

## Verification status — honest, not claimed

- `python stargate.py --selftest` → **569 checks, all passing**
  (physics pins, hostile-input probes, ledger audits).
- `python fuzz_test.py` → **11,832 hostile inputs, 0 crashes, 0 silent opens**
  — every malformed input returns a verdict, never opens the gate.
- `warp_physics.py` — the shared physics file is byte-identical with the
  Warpdrive.py copy and asserted so by both self-tests.
- What "operational" means here: every modelled system returns truthful,
  check-enforced verdicts; hardware-level items are rated TODAY/NEAR in the
  buildability ledger; the throat remains open physics (`NO FAB PATH`) —
  the twin proves exactly where engineering ends and research begins.

## Files

- `stargate.py` — physics, runtime model, reports, viewer (single file)
- `warp_physics.py` — the shared physics module (byte-identical to the
  Warpdrive copy; both self-tests assert it)
- `OVERVIEW.md` — this document; `README.md` — quick reference
- `PROJECTGOAL.md` — the objective; `PROJECTREF/` — the reference physics
- `officialgoal.md` — the living spec: every round's changes, check counts,
  coverage map (byte-identical to the Warpdrive copy)
- `fuzz_test.py` — input-fuzzing harness over the public API
- `build_state.json` — persisted build sign-offs
