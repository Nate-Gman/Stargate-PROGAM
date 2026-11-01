# Stargate Program — SG-0 "Stand-Still Gate" digital twin

`stargate.py` models a stationary stargate per `PROJECTGOAL.md`, built on the
physics of `PROJECTREF/Warpdrive.py`: a warp ring held at `v_s = 0` that anchors
a Morris–Thorne wormhole mouth. Everything is SI, everything is computed, and
every engineering barrier is priced rather than hidden. See `OVERVIEW.md` for
the full system tour.

## Run

```
python stargate.py              # 3-D interactive digital twin (pygame, default)
python stargate.py --window2d   # 2-D tkinter window (no pygame needed)
python stargate.py --console    # text console
python stargate.py --selftest   # 164 regression checks
python stargate.py --demo       # scripted end-to-end operation
python stargate.py --all        # every report
python stargate.py --json       # machine-readable export
python stargate.py --dims FILE  # JSON overrides merged into the DIMS spec
```

Reports: `--spec --metric --gate --dial --timeline --network --map --qt
--plant --ops --hurdles --deploy --buildable --fabricate --buildplan`

## The viewer

Four tabs, clickable or by key. Every number on screen is live model state —
nothing is painted on.

| Tab | Key | What it shows |
|---|---|---|
| MODEL | `M` | the rendered gate site (~30 hardware systems), data plate, ops console, gauges, component inspector (`ENTER`) |
| WORMHOLE | `W` | two-sheet embedding + traversal pulse, NEC ledger, payload-class verdicts, duty forecast, candidate sources, live crossing pipeline |
| NETWORK | `N` | all deployed links with drift + CTC onset, mouth map, site roster + notes, deploy costs, BFS routes, held-liveness check |
| BUILD | `B` | 14-step fabrication manual: dependencies, sign-off state (persisted), schedule, per-step parts/QC/crew |

Other keys: `ESC` back · `I` iris · `T` send person · `G` send signal ·
`S` service/refuel/respin · `+/-` sim speed · `SPC` pause · drag to rotate ·
click a component to inspect it.

## Architecture

| Layer | What it computes |
|---|---|
| Stationary warp field | Alcubierre f(r), York theta, T00, energy integral, Casimir lattice |
| Throat | Morris–Thorne metric, flare-out, exotic bill `-c^4 b0/6G`, tidal shear, Ford–Roman QI, Visser thin-shell, MMP |
| Timeline | mouth clocks `dtau/dt = sqrt(1+2Phi/c^2)/gamma`, desync offsets, CTC onset `|offset| = D/c`, teleport solves, frame delta-v |
| Machine | mass (555 t/gate), power/thermal/ops/buildability ledgers |
| Runtime | `Gate` / `GatePair` / `GateNetwork` — chevrons, iris, manifests, fuel, wear, service, tick/simulate/hold_live, save/load |
| Operations | dial (key\|glyph\|chevron seq), route/net transit, hold, teleport, steer, deploy, balance |

The viewer runs six live state-caches: the live pair `gp`, a probe pair `gpr`
for verdicts (keeps the live manifest clean), `netw` topology, `SIM90` (a
deterministic 90-day duty-cycle forecast), `HOLD48` (a real 48-hour liveness
hold), `PAYV`/`HOLDV`/`MMP` (per-class×source verdicts, hold bounds, monopole
summary).

## Source classes (throat-holding)

- `quantum` — Casimir lattice; Ford–Roman bound applies → hold ~1.7e-26 s,
  **zero** crossings (the field dies mid-transit)
- `mmp` — static fermion-Casimir BH pair; not sampling-bound → sendable;
  needs unobserved magnetic monopoles: 45,834 M☉, Q ~8e24 C (theoretical)
- `classical` — hypothetical non-quantum-field condensate; not sampling-bound
  → fuel-limited ~169 yr holds (an open engineering assumption)

## Physics honesty notes

- The throat's NEC ledger is real: `rho+tau = -8.6e41 J/m3` VIOLATED, ANEC
  `-4.5e42 J/m2`, exotic bill `-6.8e43 J` (~`-7.5e26 kg` of exotic mass).
- Tidal at b0 = 3.35 m is `1.6e15 g` — LETHAL; a person-safe throat needs
  `b0 ~1.35e8 m` (and a 10 m/s walk through that takes 626.7 days of
  coordinate time vs 22.2 s of free-fall proper time).
- Distance is irrelevant **only after a mouth exists** — deployment prices
  the subluminal haul (Proxima: 4.97 yr at β = 0.855, ~46 ZJ kinetic).
- Desynced wormholes are real time machines: the B-LAB↔MARS pair's CTC onset
  is ~1.5e4 yr; LEO-RELAY↔LUNA crosses in ~61 yr; Hawking's chronology
  protection conjecture is reported, not asserted.
- Exit frame-velocity mismatch is priced: a LEO arrival needs ~7.7 km/s Δv.
- Mass-flux backreaction is tracked: each person-hop moves ±80 kg between
  mouths; ~6,938 one-way crossings would zero a 555 t mouth — traffic must
  balance (MTY stable when it does).
- The `signal` mode is genuine quantum teleportation of *states*, not matter:
  a human is 22.2 Gyr of the 1e12 bit/s channel.
- Wear, not fuel, is the throat's lifetime: 0.1%/h integrity → service by
  ~day 21, collapse ~day 42 unserviced; the fuel reserve would last 169 yr.

## Key API

```
Gate(site)                    # a mouth's runtime state machine
GatePair(a, b)                # one wormhole = two mouths (shared throat)
GateNetwork()                 # link topology + BFS route-finding
gate_dial(site, payload, source, dwell_s, sheathed, phi_c2, beta)
gate_teleport(site, target_s, era)
deploy_mission(site, beta)    # pre-network haul cost
dial_window(site, max_ly)     # when a moving mouth is in reach
throat_hold_ledger(source)    # per-source maximum hold time
GatePair.ping / hold_live / balance / link_status
```

Files: `stargate.py` (the whole model + viewer), `build_state.json` (build
sign-off persistence), `fuzz_test.py` (input-fuzzing harness),
`PROJECTGOAL.md`, `PROJECTREF/` (the reference physics), `OVERVIEW.md`.
