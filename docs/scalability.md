# Scaling BeamOpt beyond PortPy lung

*Status: design proposal, 2026-11-05. Nothing in this document is implemented yet, except where it says so.*

> **Decision (2026-11-05):** finish the PortPy validation first. OpenKBP-Opt and the other datasets are listed only as future
> considerations. The point of listing them is to keep the code open to different data formats and inputs (the `Case` /
> `SiteProfile` split below), not to start adapters now.

## 1. Where we are

BeamOpt runs on **one data source**, the PortPy dataset, and **one protocol**, lung at 60 Gy in 30 fractions. Both assumptions are spread through the code:

| Coupling | Where | Count |
|---|---|---|
| Lung ROI names (`ESOPHAGUS`, `CORD`, `LUNGS_NOT_GTV`, `HEART`, `'PTV'`) | `app/web/js` (board, DVH), `src/agent/goals.py` (11), `src/agent/prompts.py` (8), `src/patient_scan.py` (7), `src/plan_session.py` (5), schemas, viz, … | ~70 lines in 15 files |
| PortPy objects (`case.plan`, `case.influence_matrix`, `case.structures`, `case.beams`, `case.ct`) | data adapter, plan session, criteria, viewer, export, scorecard, deliverability | 37 call sites |
| A single target `PTV` with a single prescription | PTV-coverage protection, criteria, the PTV-weight guard, scorecard normalisation | everywhere |
| Lung protocol only (`PROTOCOLS = {(30, 60.0): 'Lung_2Gy_30Fx'}`) | `patient_scan.py` | 1 |

The optimizer core in `plan_session.py` is generic. It needs:
- a sparse influence matrix `A` (voxels × beamlets);
- voxel volumes;
- ROI → voxel indices;
- prescription and fraction count;
- a list of objective and constraint terms.

Everything lung-specific sits at the edges: the agent prompts, guards, trade-off rows, UI tables, scorecard choice and patient scanner.

## 2. Target architecture

```
           ┌────────────── adapters (one per data source) ───────────────┐
 PortPy ──►│ PortPyCase   OpenKBPOptCase   TROTSCase   CORTCase   DicomCase│──► Case (data contract)
           └──────────────────────────────────────────────────────────────┘
                                                                              │
 configs/sites/*.yaml ──► SiteProfile (ROIs, roles, synonyms, protocol) ──────┤
                                                                              ▼
                     PlanSession (optimizer) ─► agent (prompts/guards/trade-offs from SiteProfile)
                                            └─► UI, scorecard, deliverability, export
```

### 2.1 `Case`: the data contract every adapter fills

| Field | Needed by | PortPy | OpenKBP-Opt | TROTS | CORT |
|---|---|---|---|---|---|
| `A` sparse voxels × beamlets (dose per unit intensity, per fraction) | optimizer | ✓ | ✓ `dij.csv` | ✓ per ROI (stack the matrices) | ✓ |
| voxel volumes, ROI → voxel indices, fraction of ROI inside the dose box | metrics, optimizer | ✓ | ✓ 128³ grid + `voxel_dimensions` | ✓ (ROI rows) | ✓ |
| targets: list of (name, Rx Gy), fractions | goals, criteria | 1 PTV | **3 PTVs** (70/63/56 Gy) | several | several |
| beams: per-beam beamlet grid (BEV row/col per beamlet) | smoothness term, fluence maps, leaf sequencing | ✓ | ✓ 9 beams, 64 × 64 fluence grid, `beamlet_indices` | partly (VMAT/CK: not fluence-map IMRT) | ✓ |
| CT + voxel → CT map | slice viewer, RTDOSE export | ✓ | ✓ CT on the 128³ grid | ✓ `patient` struct | ✓ |
| reference dose | ECHO-style benchmark | ECHO IMRT | clinical reference plan | `solutionX` | — |
| protocol: criteria, default objectives, scorecard | goals, criteria tab, scorecard | PortPy JSON | 10 criteria in the paper (D0.1cc / Dmean / D99) | in-file `problem` (objectives, constraints, priorities) | paper |

Each adapter is one module: `src/cases/portpy.py`, `src/cases/openkbp_opt.py`, and so on. The rest of the code uses only `Case`; no other module calls a PortPy object directly. Code that needs PortPy-only features goes through optional capabilities, so the UI can hide what a source cannot do:
- `case.capabilities = {'leaf_sequencing', 'rtdose_export', 'scorecard', ...}`
- examples: leaf sequencing (PortPy's Siochi sequencer) and the DICOM header template.

### 2.2 `SiteProfile`: everything that is lung-specific today, in data

`configs/sites/lung_60gy_30fx.yaml`, which would replace today's hard-coded constants:

```yaml
name: Lung 60 Gy / 30 fx
match: {fractions: 30, rx_gy: 60}                 # patient_scan picks the profile
targets: [{roi: PTV, rx_gy: 60, coverage_metric: D95, tolerance_gy: 0.5}]
oars:
  ESOPHAGUS:     {synonyms: [esophagus, oesophagus], guard: {Dmean: 1.0, D2: 2.0}, voxelwise: true}
  CORD:          {synonyms: [cord, spinal cord],     guard: {Dmax: 2.0},          voxelwise: true}
  HEART:         {synonyms: [heart],                 guard: {Dmean: 1.0},         voxelwise: true}
  LUNGS_NOT_GTV: {synonyms: [lung, lungs, total lung, MLD], guard: {Dmean: 1.0, V20: 2.0}, group: lung}
tradeoff_rows: [[PTV, D95], [PTV, Dmax], [ESOPHAGUS, Dmean], [CORD, Dmax], ...]
display_metrics: {PTV: [D95, D98, Dmax, V60], ESOPHAGUS: [Dmean, D2, Dmax], ...}
criteria: portpy:Lung_2Gy_30Fx
optimization_params: portpy:Lung_2Gy_30Fx
scorecard: portpy:SC_Lung(60Gy)_2022MAAS_ExampleV2.json
```

The agent prompts are built from the profile: the ROI list, the synonym lines and the examples. That way the "lower the mean lung dose" class of mistakes is handled once per site, not once per prompt.

Multi-target sites such as head and neck (PTV70/63/56) need two generalizations:
1. Protections are per target.
2. The PTV-weight guard covers every target term.

This is the only real logic change. Everything else is configuration.

### 2.3 Compute limits that grow with new data

| Limit | Today (Lung_Patient_3) | What breaks |
|---|---|---|
| Dense Gram `AᵀA` per quadratic ROI (`n_beamlets²` × 8 B) | 2,558 beamlets → 52 MB per ROI | H&N, 9 beams × up to 64 × 64 grid: 5–10k beamlets → 0.2–0.8 GB **per ROI**. Fix: keep `‖A_r x‖²` as a sparse voxel term above a size threshold, or use float32 Grams with lazy eviction. |
| Influence matrix in RAM | 0.4–1.8 GB per PortPy lung patient (planner beams) | TROTS "H&N Alt" and CORT are much larger. Fix: load one patient per process, keep `A` in float32, and memory-map the CSR arrays. |
| One solve on the CPU (Clarabel) | 20–60 s | Acceptable for one user, not for a shared server. See 2.4. |
| Constraint generation over max-dose rows | seeds about 30k rows | Fine. Fully generic. |

### 2.4 Serving: from one physicist to many

Today the app is one process with a global `APP` holding one patient, guarded by one lock.

**Step 1, already done:** sessions persist to disk (`runs/plans/<patient>/<session>/`) and can be restored after a restart.

**Step 2:** split into a stateless UI plus a **solver worker pool**:
- Each loaded patient lives in a worker process that holds the `Case` and the `PlanSession`.
- A UI session connects to a worker.
- Solves are queued.
- Workers are evicted under an LRU policy when memory runs short; the on-disk session makes that safe.

**Step 3:** the LLM is already a separate service (Ollama in Docker), so it scales on its own.

## 3. Candidate datasets

| Dataset | What it is | Why it helps | Cost to adapt |
|---|---|---|---|
| **PortPy prostate** (same HF dataset) | 129 prostate patients, protocol `Prostate_26Fx` (2.7 Gy × 26). Criteria and opt params ship with PortPy. | Second site with **zero new data format**. It tests the SiteProfile refactor in isolation, with new OARs (rectum, bladder, femurs, urethra). | Low: profile YAML plus prompts. Patients are about as large as lung (beam files 70–120 MB each). |
| **OpenKBP-Opt** ([repo](https://github.com/ababier/open-kbp-opt), [paper](https://arxiv.org/abs/2202.08303)) | 100 head-and-neck IMRT patients, the OpenKBP test set. Dose-influence matrices from CERR, 9 equidistant coplanar 6 MV beams, 128³ grid. Sparse CSV files: `dij`, `beamlet_indices`, structure masks, `possible_dose_mask`, `voxel_dimensions`. Reference plan dose plus 19 KBP-predicted doses per patient. 10.2 GB required download (OneDrive); code is MIT. | Different site, **three dose levels**, a published criteria set (cord D0.1cc ≤ 45, parotid Dmean ≤ 26, PTV70 D99 ≥ 66.5, …). Predicted doses give "what is achievable" targets per patient, a natural source of physicist-style requests ("parotids down towards the predicted dose"). | Medium: CSV adapter, multi-target goals, H&N profile, Gram memory (2.3). No ECHO plan, but the reference plan plays that role. |
| **TROTS** ([data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5387893/), [project](https://www.erasmusmc.nl/en/cancer-institute/research/projects/radiotherapy-trots)) | 120 problems: prostate CyberKnife (30), prostate VMAT (30), H&N VMAT (15, plus 15 denser), protons H&N (20), liver (10, non-convex). MATLAB v7.3 (HDF5) files with `problem` (objectives, constraints, priorities), `data` (one sparse matrix per ROI), `patient` (CT). Research use. | The richest **objective and priority definitions**, written by clinicians, plus a reference solution. A good benchmark for "does the agent's plan reach the clinical wish-list?". | Medium-high: VMAT and CyberKnife are not fluence-map IMRT, so leaf sequencing and fluence maps don't apply (capability off). Per-ROI matrices must be stacked. Protons need a different deliverability model. |
| **CORT** ([GigaScience 2014](https://academic.oup.com/gigascience/article/3/1/2047-217X-3-37/2682969)) | 4 cases: prostate, liver, H&N and the TG-119 phantom. CT, structures and dose-influence matrices (MATLAB). About 70–100 GB in total. | The TG-119 phantom is a standard QA geometry, a cheap, well-known **smoke test**. | Medium. The data is large for only 4 cases, so use it for regression tests, not validation. |
| Own DICOM via matRad or PortPy's own pipeline | Compute an influence matrix for a clinic's anonymised CT/RTSTRUCT | The only route to "the physicist's own patients" | High (dose engine, beam model). Later. |

Not useful for optimization: OpenKBP (the original challenge), GDP-HMM and other dose-*prediction* sets. They have doses but no influence matrices. They could still supply achievable-dose targets.

## 4. Proposed order

1. **Wait for validation round 1 to finish.** The batch launches new Python processes, so the agent and harness code must not change underneath it.
2. **Refactor to `Case` + `SiteProfile`, with lung as the only profile.** Acceptance: round-1 scenarios reproduce (same parses, same success on Lung_Patient_3 within repeat noise); all tests pass.
3. **Add PortPy prostate** (profile only):
   - Download about 5 small patients.
   - Run the calibration (`stage1_tools.py`).
   - Write about 10 prostate scenario templates.
   - Re-run the judge.

   This tests whether the agent generalizes to a new site with no new data code.
4. **OpenKBP-Opt adapter plus multi-target goals.** Start with about 5 patients. Use the reference plan as the benchmark and the predictions as achievable targets.
5. **Worker pool (2.4 step 2)**, once more than one person uses the app.
6. TROTS and CORT as benchmark and regression sets, with capabilities switched off where they don't apply.

## 5. Open questions for the user

- Prostate first (cheap) or OpenKBP-Opt first (the dataset you named; more different, more work)?
- Should OpenKBP-Opt's predicted doses become agent *targets*? That turns BeamOpt into a "KBP-assisted" planner, which is a different product story from "physicist instructions".
- Data storage: OpenKBP-Opt is 10 GB and CORT up to 100 GB. Is local disk enough, or should adapters stream from a network drive?
