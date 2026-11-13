# BeamOpt — physicist-instruction agent for PortPy IMRT planning

Research prototype. A medical physicist types an instruction ("reduce the esophagus mean by 2 Gy, keep PTV
coverage"). A local LLM, orchestrated by LangGraph, turns it into bounded changes to the fluence
optimization, re-solves, checks the result against deterministic dose metrics, and reports. **Not for clinical use.**

## Demo

https://github.com/user-attachments/assets/271de94c-9039-4b64-8d64-6322f164ba24

A physicist request on a lung patient: the agent changes the optimization, re-solves, checks the result against the
protocol and the organs you did not mention, and the physicist keeps or discards it.
<sub>CT data: NSCLC-Radiomics (TCIA, CC BY-NC 3.0), via the PortPy dataset. See [citations](#data-licences-and-citations).</sub>

## Layout

| Path | What it is |
|---|---|
| `src/metrics.py`, `src/criteria_evaluator.py`, `src/data_adapter.py` | M1: dose metrics, protocol-criteria screening, PortPy adapter |
| `src/plan_session.py` | M3: fast re-solvable optimizer that mirrors PortPy's `create_cvxpy_problem`, with validated knobs and version history |
| `src/agent/` | LangGraph agent (`graph.py`), structured schemas, deterministic goal checks (`goals.py`), prompts, LLM factory |
| `app/server.py` + `app/web/` | Physicist workspace (FastAPI + vanilla JS, no build step): goals board, revision timeline, linked three-plane viewer and DVH, command bar with dictation |
| `app/workspace.py`, `app/revisions.py`, `app/web_payloads.py`, `app/volume_viewer.py`, `app/viz.py` | Per-patient state behind the web app, the revision log (keep / discard / branch), JSON payloads, CT / mask / dose volumes for the viewer |
| `src/plan_export.py` | Auto-saves every solved version (fluence + settings JSON); exports RTDOSE DICOM on request |
| `src/transcribe.py` | Local speech-to-text for the mic button (faster-whisper on CPU) |
| `src/patient_scan.py` | Finds and validates PortPy patient folders in a chosen directory (files, planner beams, ROIs, prescription → protocol) |
| `src/session_report.py` | Self-contained HTML audit report of a session (requests, optimizer changes, checks, trade-offs) |
| `src/scorecard.py` | Protocol plan-quality scorecard (PortPy score curves) with per-item maxima and missing-structure handling |
| `src/deliverability.py` | Leaf sequencing (Siochi, 40 levels): segments, relative MU, fluence modulation, dose change after sequencing |
| `docs/scalability.md` | Design proposal: data adapters (OpenKBP-Opt, TROTS, CORT, PortPy prostate), site profiles, compute/serving limits |
| `experiments/` | Staged evaluation (0–5), scenario files, `report.py`; `judge/` = blind digests + rubric for LLM / human rating |
| `docker/docker-compose.yml` | Ollama with GPU in Docker |
| `configs/agent.yaml` | Data, solver, LLM, and agent settings |

## Setup

```bash
# 1. LLM server (Docker Desktop + NVIDIA GPU)
docker compose -f docker/docker-compose.yml up -d
docker exec beamopt-ollama ollama pull qwen3:8b        # also: qwen2.5:7b, llama3.1:8b

# 2. Python deps (into the existing .venv)
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 3. Patient data: Lung_Patient_3 with planner beams under portpy_quicklook_data/data/
#    (see PortPy_Single_Patient_Quicklook.ipynb, or: download_portpy_data --patients Lung_Patient_3 --beam-mode planner)

# 4. Run
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe app/server.py                # http://127.0.0.1:7861
```

## Using the workspace

- **Layout:** goals board (left), revision timeline (middle), linked viewer (right): axial, coronal and sagittal panes
  on one mm scale plus the DVH. Column widths can be dragged and are remembered; below tablet width the columns stack.
- **Goals board:** baseline, ECHO and the shown revision side by side for every protocol criterion and key metric,
  with pass / near-limit / fail marks. Each row can be locked ("keep this organ from getting worse") or set as a
  target; clicking a row selects that organ in the viewer.
- **Revisions:** each request becomes a card with its outcome, the changes, the result against its parent, the cost
  to other organs and what blocked it. **Keep**, **Discard** (back to the parent plan) and **Branch** (continue from
  this revision). Sending a new request keeps a revision that is still waiting for review ("build on it").
  A result marked **needs your decision** met the goal only at a cost the agent will not accept on its own.
- **Stop:** while the agent works, **Stop** ends the request after the current step; the plan goes back to where the
  request started.
- **DVH:** solid = shown revision, dashed = its parent, dotted = ECHO. Hover for a readout, click a curve or legend
  entry to select or hide it, drag a box to zoom (double-click resets). Goal and limit markers can be dragged; this
  fills an instruction into the command bar but never sends it.
- **Viewer:** CT window presets, dose wash, isodose lines, contours and a difference map against the parent revision.
  Volumes are fetched once and cached. The mouse wheel changes slices; clicking moves the shared crosshair.
- **Why this dose:** for the selected organ, how much each beam contributes and which hard limits the plan is pressed
  against (ranked by how much they cost the objective).
- **Storage:** every solved version is saved to `runs/plans/<patient>/<session>/` as `v<N>.json` (knobs, changes,
  metrics) plus `v<N>_fluence.npy`. **Export DICOM** (toolbar) downloads one zip for the shown plan. The zip contains an
  RTDOSE on the CT grid in total Gy, using the dataset's frame of reference (checked by a round trip through PortPy's
  reader: max error 4e-6 Gy), plus the plan JSON, the fluence and a README. A copy is kept under `runs/exports/`. No
  RTPLAN is written, because a deliverable plan needs leaf sequencing.
- **Patient:** open **Settings**, type or **Browse…** to a data folder (the folder holding `Lung_Patient_*/`, or a download root with `data/`), pick a patient and **Load patient**. The scanner checks the PortPy files and the influence matrix of every planner beam, reads the ROIs and the ECHO RTPLAN prescription, and only offers protocols the app supports (Lung_2Gy_30Fx). Loading frees the previous patient; the first load of a patient also solves its baseline (cached after that). Patients without calibration hints still work, with generic guidance (run `experiments/stage1_tools.py --patients <id>` to calibrate, ~7 min).
- **Trade-offs:** every reply that changed the plan ends with a before → after table across PTV and all OARs (✅ better, ⚠️ worse, · no real change).
- **Saved sessions:** in the same panel, **Saved sessions folder** (default `runs/plans`, or **Browse…** to any folder holding sessions, one patient's sessions, or a single session) lists previous sessions, the loaded patient first. **Restore session** loads that patient if needed, recomputes every version's dose from the saved fluence, and restores the current plan, undo stack, chat and request log (`session.json`). New requests continue in the same folder. Sessions with only the baseline are not written.
- **Scorecard** (Details drawer): the protocol plan-quality score (`SC_Lung(60Gy)_2022MAAS`) of the viewed plan, the compared plan, the baseline and ECHO, per item. It also shows a total after normalising the dose to PTV D95 = Rx, because unnormalised plans lose the PTV V60 points. Items for structures the patient doesn't have are listed as not scored and are left out of the maximum.
- **Deliverability:** leaf-sequences the viewed plan (step-and-shoot, 40 levels) and shows segments and relative MU per beam against the baseline (or the compared plan), fluence modulation, how much sequencing changes key metrics, and beam's-eye-view fluence maps (or the difference map when comparing). This is a screening check: MU are in fluence units, with no machine model.
- **Session report:** **Report** (toolbar) saves an audit report: each request, the optimizer changes, the deterministic checks, the trade-offs and the agent's reply (quoted, labelled as narrative). The export zip includes it too.
- **Model:** the local language model is picked in Settings (qwen3:8b is the validated default).
- **Dictation:** press the microphone button, speak, then press it again. The transcript appears in the command bar so you can edit it before
  sending. The Whisper model (`asr.model` in `configs/agent.yaml`, default `base.en`) downloads once, about 150 MB.

## How the optimizer stays fast and exact

PortPy's lung problem has 432,753 voxels × 2,558 beamlets. Solving it as PortPy builds it took about 180 s with
Clarabel on this machine. `PlanSession` solves the **same** problem much faster:

- **Gram matrices.** Every `quadratic` and `smoothness` term is `‖Mx‖² = xᵀ(MᵀM)x`, so the 2558² Gram matrices
  are precomputed once.
- **Shared dose variable.** One `d_PTV = A_PTV x` is shared by the PTV over- and underdose terms and by the PTV
  max constraint.
- **Constraint generation.** Max-dose rows are added only when they are near-active. Every solve is checked
  against all voxels; the tolerance is 0.05 Gy, and the worst excess is recorded per version.

Parity check (Stage 0): at our solution, PortPy's own objective matches ours to a relative difference of 2e-9,
with zero PortPy constraint violations. Typical re-solve time is 20–50 s. Optionally, `merge_x: 2` merges
beamlet pairs (5 mm) for about 10–20 s solves, at a small quality cost.

## Agent design (LangGraph)

```
understand ─┬─(optimize)─► propose ─► apply(validate) ─► optimize ─► evaluate ─┬─► finalize ─► report
            │                 ▲                                               │
            │                 └──────────────── iterate (≤4) ◄────────────────┘
            ├─(question)─► answer         ├─(revert/reset)─► restore
            └─(out of scope / unsafe / unavailable ROI / already met)─► refuse
```

**What the LLM does**
- Parses the request into a structured `PhysicistGoal` (ROI, metric, direction, amount/target, protections).
- Proposes 1–3 knob changes:
  - set a weight;
  - add a `mean`, `quadratic` or `linear_overdose` objective;
  - add or tighten a `max` or `mean` constraint.
- Writes the final reply.

**What deterministic code does**
- Resolves the goal into numeric checks. Default protections are PTV D95 ≥ current − 0.5 Gy and PTV Dmax ≤ current + 1 Gy.
- Adds **side-effect guards** for every OAR the physicist did not mention: ESOPHAGUS Dmean +1 / D2 +2 Gy, CORD Dmax +2 Gy, HEART Dmean +1 Gy, total lung Dmean +1 Gy / V20 +2 %-pt (`agent.oar_guard` in `configs/agent.yaml`). An attempt that breaks a guard is rolled back like one that breaks a protection.
- Requires each attempt to change at least one knob of the organ the request is about (another organ may be changed in addition, e.g. to protect it).
- Validates every action:
  - unknown ROIs, and ROIs with no voxels in the calculation box (HEART here);
  - relaxing protocol limits;
  - lowering PTV coverage weights.
- Solves, and judges pass or fail.
- Chooses the best attempt that keeps the protections. If none does, it restores the plan.

**Data caveats for Lung_Patient_3**

The calculation box covers only part of some organs:

| ROI | Coverage |
|---|---|
| ESOPHAGUS | 56% |
| CORD | 41% |
| LUNGS | ~50% |
| HEART | 0 optimization voxels |

Metrics for these ROIs describe the represented sub-volume and are labeled that way. The conservative protocol
criteria table marks them UNDETERMINED.

## Experiments

| Stage | Script | What it measures |
|---|---|---|
| 0 | `stage0_infra.py` | Load and solve times, PortPy parity, re-solve time, LLM tokens/s |
| 1 | `stage1_tools.py` | Scripted knob changes move metrics as expected (no LLM); sensitivity table |
| 2 | `stage2_intent.py` | Instruction → structured goal accuracy on 30 labeled requests, per model |
| 3 | `run_experiments.py --stage 3` | End-to-end achievable requests, judged by harness checks and protections |
| 4 | `run_experiments.py --stage 4` | Safety and honesty: unavailable ROI, unsafe/relax/prescription requests, impossible goals, multi-turn, undo, factual answers |
| 5 | `make_scenarios.py` + `run_experiments.py --scenarios scenarios_echo.yaml` | Targets taken from the ECHO benchmark plan (same 7 beams) |

`python experiments/report.py` writes `experiments/results/REPORT.md`.

### Multi-patient validation

- `scenarios_multi.yaml` holds patient-independent templates. Each is rendered from the patient's own baseline (e.g. "reduce the esophagus mean by 15 % of its value, 0.5–2 Gy") and skipped when it does not apply (e.g. no heart, esophagus already < 3 Gy).
- `stage1_tools.py --patients ...` calibrates each patient (scripted knob changes → `runs/cache/hints_<id>.txt`).
- `run_experiments.py --scenarios scenarios_multi.yaml --patients ...` runs the agent. Success requires the goal, the PTV protections, no new protocol FAIL **and** the side-effect guards (harness's own copy of the tolerances); `success_no_guard` keeps the old definition for comparison.
- `run_validation.sh` is the first-round batch: 10 held-out lung patients × 2 repeats, plus the development patient (Lung_Patient_3, where prompts were tuned; reported separately).
- `judge/make_digest.py <run.jsonl>` writes blind digests (no pass/fail label) for rating with `judge/rubric.md`; ratings go to `results/judge/ratings_*.jsonl` and `report.py` joins them with the harness results.

### Evaluation data from the PortPy dataset

**Used**
- **ECHO reference IMRT dose and plan**: `rt_dose_echo_imrt.dcm` and `rt_plan_echo_imrt.dcm`, available for 330/331
  patients. It uses the same planner beams as PortPy.
- **PortPy clinical criteria**: `Lung_2Gy_30Fx`.
- **Lung plan-quality scorecard**: `SC_Lung(60Gy)_2022MAAS`, scored with `Evaluation.compute_total_quality_score`.
- **Per-ROI calculation-box coverage.**

**Not useful for scoring**
- `data_info.jsonl`: a file index only.
- The TCIA metadata.
- The VMAT plans: prostate only.

## Data, licences and citations

BeamOpt is a non-commercial research prototype. It depends on PortPy and on PortPy's patient data, and both are
licensed for non-commercial use only.

| Source | Licence | What it means here |
|---|---|---|
| [PortPy](https://github.com/PortPy-Project/PortPy) (Python package; `PlanSession` reproduces its optimization problem) | Apache 2.0 with the Commons Clause | Free for non-commercial academic and research use. The Commons Clause forbids selling a product or service whose value derives substantially from PortPy; commercial use needs a separate licence from the PortPy authors. |
| [PortPy dataset](https://huggingface.co/datasets/PortPy-Project/PortPy_Dataset) (influence matrices, ECHO plans) | CC BY-NC 4.0 | Attribution required, non-commercial use only. |
| [NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) on TCIA (the lung CT and structures behind the PortPy lung patients) | CC BY-NC 3.0 | Attribution with the data citation and its DOI is required, non-commercial use only. This also covers the CT images in screenshots and the demo video. |

No patient data is stored in this repository: download it from the PortPy dataset (see Setup).

If you use BeamOpt or show results made with it, please cite the sources below.

**PortPy and the PortPy.AI benchmark**

> Jhanwar G, Tefagh M, Huang Q, Golkarieh A, Hong L, Pham H, Taasti VT, Tuomaala S, Magliari A, Folkerts M, Zhou Y,
> Yang J, Nadeem S, Zarepisheh M. PortPy.AI: A Benchmark for AI and Optimization in Cancer Radiotherapy Planning.
> Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track, 2026.

```bibtex
@inproceedings{jhanwar2026portpyai,
  title     = {PortPy.AI: A Benchmark for AI and Optimization in Cancer Radiotherapy Planning},
  author    = {Jhanwar, Gourav and Tefagh, Mojtaba and Huang, Qijie and Golkarieh, Alireza and Hong, Linda and
               Pham, Hai and Taasti, Vicki Trier and Tuomaala, Seppo and Magliari, Anthony and Folkerts, Michael and
               Zhou, Ying and Yang, Jie and Nadeem, Saad and Zarepisheh, Masoud},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

**NSCLC-Radiomics data** (TCIA data citation)

> Aerts HJWL, Wee L, Rios Velazquez E, Leijenaar RTH, Parmar C, Grossmann P, Carvalho S, Bussink J, Monshouwer R,
> Haibe-Kains B, Rietveld D, Hoebers F, Rietbergen MM, Leemans CR, Dekker A, Quackenbush J, Gillies RJ, Lambin P.
> Data From NSCLC-Radiomics (version 4) [Data set]. The Cancer Imaging Archive; 2014.
> https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI

**NSCLC-Radiomics publication**

> Aerts HJWL, Velazquez ER, Leijenaar RTH, Parmar C, Grossmann P, Carvalho S, Bussink J, Monshouwer R, Haibe-Kains B,
> Rietveld D, Hoebers F, Rietbergen MM, Leemans CR, Dekker A, Quackenbush J, Gillies RJ, Lambin P. Decoding tumour
> phenotype by noninvasive imaging using a quantitative radiomics approach. Nature Communications 5, 4006 (2014).
> https://doi.org/10.1038/ncomms5006

**The Cancer Imaging Archive**

> Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L,
> Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of
> Digital Imaging 26(6), 1045–1057 (2013). https://doi.org/10.1007/s10278-013-9622-7

BeamOpt is not affiliated with or endorsed by the PortPy authors, MSKCC or TCIA.
