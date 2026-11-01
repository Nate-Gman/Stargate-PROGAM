# SG-0 Stand-Still Gate — System Overview

A physics-honest digital twin of a stationary stargate: a 9.8 m warp ring held
at `v_s = 0` anchoring one mouth of a Morris–Thorne wormhole, with a live
operational model (fuel, wear, manifests, service), a gate network of deployed
mouths, chronology bookkeeping, and a fabrication manual that stops honestly
where the physics runs out.

One file — `stargate.py` (~7,600 lines) — contains the physics, the runtime
model, the reports, and three front-ends: a pygame 3-D viewer (default), a
tkinter 2-D fallback (`--window2d`), and a text console (`--console`).

## The honest verdict

The machine is two different objects with two different verdicts:

- **The housing is real engineering.** A 555 t ring of Inconel 718, 27 REBCO
  coils at 19 kA × 9,600 turns held at 18 K, a 4.5 MWth D-He3 reactor
  (×0.62 direct conversion → 2.79 MWe), a 236 MJ SMES buffer riding through
  1.9 min of transient, 6 radiator sails at 985 K against a 1900 K limit —
  all aerospace-manufacturable to stated tolerances, all in the BOM.
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
- Mass-flux backreaction: each crossing moves mass between mouths; the pair
  tracks net flux vs the 555 t instability budget.

### Machinery & operations
- `Gate`: chevron sector locks (9 sectors × 40°), iris firewall (matter
  admitted only when open; inbound `receive()` honors it too), fuel, wear
  (0.1%/h → service ~41 d, collapse ~42 d), manifests, logs, persistence.
- `GatePair`: shared throat, `link_status` (both ends open + clock
  antisymmetry), `ping` (2.7 s throat RTT vs 4.4 min light-time ≈ 98×),
  `hold_live` (periodic re-verification — a held link dies with the plant,
  not silently), `balance` (mass-flux drift vs the 5% destabilization
  threshold), `transit` (books both manifests, stamps receiver-clock
  arrivals).
- `GateNetwork`: 9 registered sites, 5 deployed mouth-pairs, BFS routes
  through multi-mouth sites (EARTH-FAR/LEO/LUNA/MARS host 2 mouths each),
  deploy-mission costs for mouth-less sites (a wormhole cannot be dialed
  where no mouth is — Proxima is a 4.97 yr haul at β = 0.855).
- Addressing: 38-glyph alphabet, chevron sequences (MARS =
  `0-0-0-0-0-0-0-4 +4@origin`), `resolve_dial` site lookup.
- Duty-cycle `simulate()`: deterministic wear/fuel forecast — integrity
  warnings at days 20.8/31.3/37.5, collapse at 41.7 d; fuel is never the
  limiter.

## The four screens

**MODEL [M]** — the gate site rendered as ~30 hardware systems (torus shell,
collars, brace cage, cryostat shroud, 27 coil packs, 9 chevron wedges with
glyphs/IDs/pistons, 12-blade iris, horizon ring, pedestal/saddles/pad,
reactor+SMES tori, fuel tank with live level, comms dish, temperature-tinted
radiators, mouth B on its wormhole arc with the <1 ns sync pulse). Data
plate, ops console (link/ping/fuel/integrity/service/power/log lines),
gauges, and a click-to-inspect component system (`ENTER` for the full spec
page: material, geometry, qty, process, mass, power, duty, failure mode,
installs-step).

**WORMHOLE [W]** — the physics page. Left: the animated two-sheet embedding
with anatomy callouts and a descending traversal pulse. Center: the 16-row
throat ledger, the payload-class table (5 classes × quantum/mmp verdicts —
vehicle is tidal-blocked at any source, needing `b0 = 2.7e8 m`), the MMP
two-clocks fact (626.7-day coordinate walk vs 22.2 s free-fall, 626.7-day
air bill), and the deterministic 90-day duty forecast to collapse. Right: QI
log-bars, three live transit verdicts (person[mmp] sendable, signal[quantum]
QI-blocked, person+sheath[classical] sendable with a 2e15 g pod shell),
CTC status + onset, the 5-row candidate-source table, hold bounds per source,
drift cause (Φ/c²), strobe duty, the QT channel, the teleport dwell/Δv/CPC
line, a 6-stage live crossing pipeline, and the mass-flux ledger with the
one-way-zeroing budget.

**NETWORK [N]** — all 5 deployed links with separation, real drift/day, and
CTC onset (LEO-RELAY↔LUNA: 61 yr; MARS↔HELIOTAIL: 1,718 yr; B-LAB↔EARTH-FAR:
never). Log-radius mouth map with packed labels and the live arc. Site roster
(addr, separation, deployed, mouths, carry-years, drift vs pad) plus each
site's registry note. Deploy costs for undeployed sites, BFS route printouts,
and the 48-hour held-liveness check result.

**BUILD [B]** — the 14-step fabrication manual: procedure, sub-assembly
checklists, consumes/crew/QC/duration per step, dependency gates (`SPC` to
sign off, persisted to `build_state.json`), installed-mass tracking, and
serial vs ~40-month critical-path schedule. Step 14 (throat) is honestly
unmarkable — no procedure exists.

## Numbers that matter

| Quantity | Value | Meaning |
|---|---|---|
| throat b0 | 3.35 m | engineered mouth radius |
| rho+tau | −8.6e41 J/m³ | NEC violation at the wall |
| exotic bill | −6.8e43 J ≈ −7.5e26 kg | what holding it costs |
| QI hold | 1.72e-26 s | Ford–Roman bound (1 s pulse) |
| crossing | 1.34 s | needs 7.8e25× more hold |
| tidal at b0 | 1.6e15 g | lethal — safe b0 ≈ 1.35e8 m |
| exotic source | none priced | Casimir −69.9 kJ, 39 orders short |
| hold bound | quantum 1.7e-26 s / non-quantum 169 yr | field vs plant |
| mouth carry | Proxima 4.97 yr @ β 0.855 | distance is NOT irrelevant first |
| human teleport | 22.2 Gyr of channel | signals only, honestly |
| duty life | collapse ~41.7 d unserviced | wear kills before fuel does |

## Files

- `stargate.py` — physics, runtime model, reports, viewer (single file)
- `OVERVIEW.md` — this document; `README.md` — quick reference
- `PROJECTGOAL.md` — the objective; `PROJECTREF/` — the reference physics
- `build_state.json` — persisted build sign-offs
- `fuzz_test.py` — input-fuzzing harness over the public API

`python stargate.py --selftest` → **164 checks, all passing.**
