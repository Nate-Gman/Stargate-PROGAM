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
python stargate.py --selftest   # 569 regression checks
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
| Ephemeris | eccentric Kepler orbits (Mars e=0.0934), lunar nodal regression, retarded light-time, conjunction blackouts, pass windows with az/el slew + keyholes, aberration, Doppler |
| Link | derived G/T from billed dishes, FSPL, atmo+iono (f⁻²) delay, derived pointing loss off HPBW, ITU-R rain fade, Shannon ceiling, weather availability, mouth-to-mouth mesh relay |
| Station-keeping | per-orbit-kind dv/yr: LEO drag 15.5, MARS trim 2.0, halo 3.5, LUNA 0 |
| Civil | foundation bearing (0.17 MPa, soil binding), seismic overturn/slide, ring thermal breathing 3.0 mm/day |
| Machine | mass (557 t/gate over 15 billed lines), power/thermal/ops/buildability ledgers, parts book + DIMS glossary (89 keys, all defined) |
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
  mouths; ~6,963 one-way crossings would zero a 557 t mouth — traffic must
  balance (MTY stable when it does).
- The `signal` mode is genuine quantum teleportation of *states*, not matter:
  a human is 22.2 Gyr of the 1e12 bit/s channel.
- Wear, not fuel, is the throat's lifetime: 0.1%/h integrity → service by
  ~day 21, collapse ~day 42 unserviced; the fuel reserve would last 169 yr.
- The comms layer is real RF: Mars opposition books 937 kbps / ~5 GB/sol
  through real rise/set windows (not a fictional 24/7 link), rain fade and
  wind-stow derate availability to 99.3%, and the mesh relay admits its
  thinnest hop honestly (~10 kbps mouth-to-mouth vs the 111 GB pad leg).
- The pad is a civil structure: 0.17 MPa bearing (soil binds at 1.76×),
  1.6× seismic overturn margin, 3.0 mm of diurnal ring breathing.
- Every billed part is spec'd (material + purpose + ops + buildability)
  and every billed parameter is defined (89 DIMS+PLANT glossary keys) —
  an unglossed or unspecified line is a failing check.

## Math proofs — every headline number, derived

Each block: the published equation, the value the twin computes, and the
`--selftest` check that pins it. Nothing is typed for the story.

### 1. The throat's exotic bill (Morris–Thorne 1988)

```
E_exotic = -c^4 b0 / 6G   (Einstein eqn at the Visser thin-shell limit)
b0 = 3.35 m  ->  -6.8e43 J  =  -7.5e26 kg of exotic mass
```
[Self-test: `exotic bill ~2e43 J per metre` — PASS]

### 2. Ford–Roman quantum-inequality hold (1995)

```
tau_max ~ hbar c^2 / (720 pi^2 |rho| L^2)    rho = -8.6e41 J/m3
-> 1.72e-26 s of hold vs a 1.34 s crossing  -- the field dies mid-transit
```
[Self-test: `QI hold << crossing for spec'd gate` — PASS]

### 3. Mouth clock rate (weak-field GR + SR)

```
dtau/dt = sqrt(1 + 2Phi/c^2) / gamma
LEO mouth (orbital speed + shallower well): -24.5 us/day mean drift
(the same sign and magnitude as the real ISS correction)
```
[Self-test: `site rate: LEO mean drift ~ -25 us/day` — PASS]

### 4. CTC onset — the time-machine boundary

```
a wormhole becomes a CTC when |desync| >= separation / c
B-LAB <-> MARS onset ~ 1.5e4 yr;  LEO <-> LUNA ~ 61 yr
```
[Self-test: `CTC boundary exact: offset = D/c` — PASS]

### 5. Antenna G/T from the billed dish (not a typed constant)

```
G/T = 10 log10( eta (pi D f / c)^2 ) - 10 log10(T_sys)
D = 34 m pad dish, X-band, T_sys = 20 K -> 53.9 dB/K  (real DSN class)
```
[Self-test: `antenna: 34-m DSN-class pad lands ~54 dB/K` — PASS]

### 6. Beamwidth + derived pointing loss

```
HPBW = 70 lambda / D          34 m -> 0.0735 deg
L_pt = 12 (theta_err / HPBW)^2 -> 0.227 dB   (replaces the old typed 3 dB)
```
[Self-test: `beamwidth: 34-m pad beams ~0.07 deg`, `pointing loss is DERIVED` — PASS]

### 7. The Mars link, end to end (real DSN worksheet)

```
FSPL = 20 log10(4 pi d / lambda)  -> 266.4 dB at opposition
prx = ptx + Gt + Gr - FSPL - L_pt - L_atmo - L_rain  -> -157.3 dBW
Shannon: C = B log2(1 + SNR)     -> 937,427 bps
```
[Self-test: `Mars FSPL ~266 dB`, `real Mars-DTE class` — PASS]

### 8. Rain fade + availability (ITU-R P.838)

```
A_rain = k R^alpha L_slant(el)    -> 0.63 dB at 15 mm/h
availability = clear time - wind stow - fade outages -> 99.3%
```
[Self-test: `X-band rain fade is tenths of a dB`, `desert pad lands DSN-class availability` — PASS]

### 9. Civil structure — the gate stands on soil

```
bearing  P = m g / A  = 557 t / 32 m^2  -> 0.171 MPa
soil allowable ~0.3 MPa -> margin 1.76x  (concrete: 176x -- soil binds)
seismic: 0.3 g -> 1.64 MN lateral;  overturn 1.6x < slide 2.0x
ring breathing: dL = CTE D dT  = 3.0 mm over the diurnal swing
```
[Self-test: `foundation` x2, `seismic` x2, `ring thermal` x2 — PASS]

### 10. Station-keeping from the decay rate (LEO)

```
dv/yr = |da/dt| (v / 2a) yr      decay 2.24 km/month at the billed orbit
-> 15.5 m/s/yr   (ISS-class; LUNA 0, MARS 2.0, halo 3.5)
```
[Self-test: `reboost ledger ~5-20 m/s/yr`, `stationkeeping` x5 — PASS]

### 11. Transit dv — the arrival inherits the mouth's frame

```
dv_arrive = |v_mouth - v_site| evaluated on the real ephemeris
pad->LEO 7379 m/s,  pad->MARS 4691 m/s,  pad->LUNA 678 m/s
```
[Self-test: `transit dv` x4, `transit energy` x3 — PASS]

### 12. Mesh relay — capacity is the thinnest hop

```
C_route = min( link_hop(dest->relay), link_leg(relay->pad) )
MARS->LEO hop (5-m dishes both ends): ~10 kbps -> 0.11 GB
LEO->pad leg: 111 GB -- the hop binds; relays buy coverage not aperture
```
[Self-test: `mesh` x5 — PASS]

### 13. Wear vs fuel — what actually kills the throat

```
integrity -0.1%/h -> service ~day 21, collapse ~day 41.7 unserviced
fuel reserve at hold rate -> ~169 yr   (wear is the lifetime, not fuel)
```
[Self-test: `mmp hold = fuel-limited`, `gate hold`/`gate service` — PASS]

### 14. Deployment honesty — distance is NOT irrelevant first

```
Hohmann-class haul before any mouth exists: Earth->Mars 256 d, 5.84 km/s,
2.14-yr synodic cadence -- the wormhole's promise starts AFTER the tow
```
[Self-test: `hohmann` x5, `deploy mission` — PASS]

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
# the comms + ops layer
antenna_gt_dbk(dish_m, f_hz, tsys_k)      # derived G/T
antenna_beamwidth_deg(dish_m, f_hz)       # HPBW = 70 lambda/D
link_budget_db(site, t, f, tx, rain_mmph) # the DSN worksheet
link_weather_db(el, rain_mmph)            # ITU-R rain fade
pad_weather_ledger()                      # availability book
gate_pass_windows(site, t0, span)         # real rise/set + keyholes
gate_pass_downlink_gb(site, t0, days)     # scheduled + expected GB
gate_mesh_downlink_gb(dest, via, t0, days)# relayed capacity
mouth_stationkeeping_ledger(site)         # annual dv bill
pad_foundation_ledger() / pad_seismic_ledger() / pad_ring_thermal_mm()
gate_transit_dv / gate_transit_energy     # the crossing's real bills
gate_parts_book() / gate_dims_glossary()  # the spec book
site_hardware_spec(site)                  # a remote site as a machine
```

Files: `stargate.py` (the whole model + viewer), `warp_physics.py` (shared
physics, byte-identical to the Warpdrive copy), `build_state.json` (build
sign-off persistence), `fuzz_test.py` (input-fuzzing harness),
`PROJECTGOAL.md`, `PROJECTREF/` (the reference physics), `OVERVIEW.md`,
`officialgoal.md` (the living spec + coverage map).
