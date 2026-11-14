# Raumdeuters — Phase 1: Ball Possession

Technical document for our submission `Raumdeuters.csv`
(repository: https://github.com/stefrazv/more-than-a-hack-2027).

## 1. Summary

We label every frame of BAR-ATH as `H`, `A`, `D` or `X` from the players' trajectories, with the
ball used only as a secondary signal. The pipeline has three parts:

1. **Dead ball (`D`)**: a gradient-boosting classifier on player-only features (speeds, spread,
   formation), trained on external matches that we first degraded to look like the competition's
   tracking, decoded into stoppages with a two-state hidden Markov model (Viterbi).
2. **Team in possession (`H`/`A`)**: PathCRF (Kim et al., KDD 2026), a pretrained model that infers
   possession from player trajectories alone, corrected by a ball vote where the ball is clearly
   with one team.
3. **Possession rules**: minimum possession length, restart handling, and half boundaries.

Nothing is trained, tuned or checked on the competition match beyond format validation. Every
parameter was chosen on external data (Section 4).

## 2. Approach, and why

### 2.1 What we started with: rules on the ball (dropped)

Our first versions (V1–V5; the last one is kept in `legacy/`) were hand-written
rules: the team of the player nearest to the ball, with hysteresis (a change has to hold for
25 frames), plus a conservative dead-ball detector based on ball visibility, ball position
and restarts from the smart tagging.

We dropped this approach for two reasons:

- **The ball is often missing.** In the competition ATD it is absent on roughly 41% of frames, so
  any rule built on the ball has nothing to say for long stretches.
- **It does not survive realistic tracking.** On external matches degraded to resemble the
  competition data (Section 4.2), the ball rules scored spellF1 0.18–0.30, against ≈0.45 for the
  pipeline below.

### 2.2 Why trajectories: PathCRF

PathCRF (Kim et al., KDD 2026) infers the "possession path" (which player holds the ball, and when
it passes or is lost) from player positions and velocities only, with a conditional random field on
top of a set-based neural encoder. It was designed for exactly our situation: no reliable ball.
The authors publish the code and models pretrained on the public DFL/IDSSE matches. We use their best
model (`saved/140`, Dynamic Masked CRF) **unchanged**; it is vendored in `third_party/pathcrf/`
(MPL-2.0, see `SOURCE.md`).

PathCRF expects 11 identified players per team with no gaps. The competition ATD instead has 46
anonymous slots that appear, disappear and sometimes duplicate a player. We bridge this with
**stitching** (`src/stitch.py`). For each team and half, we turn the slots into 11 continuous
"virtual players":

- **Continuity:** a slot keeps its virtual player while it stays visible.
- **Duplicates:** a slot within 1 m of an already-assigned player is ignored.
- **Joining:** a new slot joins the free virtual player nearest to its last known position. The
  allowed distance is $5\,\text{m} + 7\,\text{m/s} \cdot \Delta t$, where $\Delta t$ is how long
  the player was unseen. A join beyond that limit is a last resort and is counted as "forced".
- **Gaps:** filled by linear interpolation.

Stitching is deterministic. On BAR-ATH the virtual players are directly observed on 90–95% of
frames, with 26–44 forced joins per team and half.

### 2.3 Dead ball from players, not from the ball

PathCRF needs to know where play is alive (it runs per "episode" of live play), and the metric
scores `D` directly. We predict dead ball with a `HistGradientBoostingClassifier` (scikit-learn;
300 iterations, learning rate 0.05) on 22 features per frame, computed from players only:

- mean, median and 90th-percentile speed, and the share of players below 1 m/s;
- each team's centroid distance from the halfway line, and its spread in $x$ and $y$;
- the distance between the two team centroids;
- rolling means over 1, 3 and 6 s, plus the trend between the past and the next 2 s.

**The ball is excluded on purpose.** In the DFL training data the ball position freezes exactly
when play stops. A model with ball features reached 0.999 accuracy on DFL by learning that artefact,
which does not exist in the competition ATD.

**From probabilities to stoppages: an HMM.** The classifier gives a probability $p_t$ that play is
dead at frame $t$. Instead of thresholding it, we decode the most likely sequence of states
$s_t \in \{\text{alive}, \text{dead}\}$ with the Viterbi algorithm (`dead_dfl.viterbi_dead`), per half:

$$\hat s = \arg\max_{s} \sum_t \Big[\, w \log b_{s_t}(p_t) + \log P(s_t \mid s_{t-1}) \Big],
\qquad b_\text{dead}(p) = p,\; b_\text{alive}(p) = 1 - p,$$

$$P(s_t = s_{t-1}) = 1 - \frac{1}{L_{s}}, \qquad P(s_t \neq s_{t-1}) = \frac{1}{L_{s}},$$

where $L_\text{dead} = 25.26$ s and $L_\text{alive} = 31.35$ s are the mean lengths of stoppages
and live stretches measured on the 7 DFL matches, and $w = 0.2$ discounts the evidence of each
frame (consecutive frames are strongly correlated, not independent observations). A stoppage is
opened only when the dead-ball evidence outweighs the cost of switching state twice, so short
confident stoppages are kept and long hesitant ones are not. This replaced our first rule (threshold
0.5, then drop stoppages shorter than 4 s and restarts shorter than 2 s); see Section 4.3.

### 2.4 Ball vote on top of PathCRF

Where the ball *is* visible, it is strong evidence. We call a frame **ball evidence** for a team
when that team's nearest player is within 2 m of the ball, and at least 0.75 m closer than the
nearest opponent. For each possession PathCRF predicts, if at least 25 frames (1 s) carry ball
evidence and at least 60% of them point to the other team, the whole possession is reassigned to
that team. On BAR-ATH, ball evidence exists on 43% of live frames.

## 3. Assets used and dropped

| asset | used? | how |
|---|---|---|
| `tactical/BAR-ATH_tactical_atd.txt` + `.xml` (tracking) | **yes** | player slots → stitching → D model and PathCRF; ball → ball vote only |
| `tactical/BAR-ATH_tactical_smart_tagging.xml` | dropped | used by the rule-based versions for restart anchors; not in the final pipeline |
| `tactical/BAR-ATH_tactical.mp4`, homography | no | |
| `broadcast/*` | no | not synchronised with the tactical timeline |
| public 5-minute ground truth | format checks only | never used for training or for choosing parameters |
| full-match ground truth | no | not available to participants |

## 4. Training data

**None of it comes from the provided game.**

### 4.1 Dataset

**DFL/IDSSE open dataset.** Bassek et al. (2025): seven complete matches of the 2022/23 Bundesliga
and 2. Bundesliga, CC BY 4.0, https://doi.org/10.6084/m9.figshare.28196177. Position data for all
players and the ball at 25 Hz, recorded with TRACAB optical tracking by Sportec Solutions on behalf
of the Deutsche Fußball Liga (DFL), the data owner. The position data carries, for every frame,
the official **ball status** (alive / dead) and **ball-owning team**. These are the only labels we
use. They come with the dataset; we did not label anything ourselves.

We preprocess the matches with PathCRF's own `datatools/preprocess.py`.

| model | trained by | data | labels used |
|---|---|---|---|
| PathCRF `saved/140` | the PathCRF authors | DFL/IDSSE matches (their split) | possession events derived from the DFL data |
| our dead-ball model | us (`src/train_dead_model.py`) | all 7 DFL/IDSSE matches, degraded and stitched | DFL ball status (dead / alive) |

The dead-ball model is trained on one frame in five (5 Hz), with a fixed seed. It is saved as
`trained_models/dead_ball_model.joblib`, so reproducing the submission does not need the DFL data.

### 4.2 Making DFL look like the competition tracking

DFL tracking is clean: 22 identified players and a ball that is almost always present. The
competition ATD is not, so a model trained on clean DFL would be trained on the wrong input.
`src/degrade.py` breaks the DFL tracking in the ways we observed in the ATD:

- each player is cut into fragments (mean 90 s) separated by gaps (mean 6 s);
- fragments are placed into 46 anonymous slots in order of appearance;
- 40% of fragments get a short duplicate offset by 0.5–2 m;
- positions get noise (0.10 m for players, 0.20 m for the ball);
- the ball is removed on 41% of frames, in gaps.

The degraded match is then stitched back exactly as BAR-ATH is (Section 2.2). The D model is
trained on this, and every parameter of the pipeline was chosen on it.

### 4.3 How parameters were chosen

We hold out DFL match `J03WR9`, train the D model on the other six, degrade everything with seed 47,
and score the full pipeline with the **official metric** (`evaluation/possession_metrics.py`) on
the held-out match. The parameters below were chosen this way, and the final D model was then
retrained on all seven matches. The experiments are in `src/` and take the folder of preprocessed
DFL matches as `--dfl`: `check_dfl_ball.py` (ball leak), `dead_dfl.py` (D features and smoothing),
`pipeline_dfl.py` (clean tracking), `robust_dfl.py` (degraded tracking), `ha_fusion_dfl.py`
(H/A methods, ball distance and $\tau$) and `hmm_dfl.py` (rule vs. HMM for D). The last two were
repeated with degradation seeds 47, 48 and 49, and we report the mean.

| parameter | value | choices compared |
|---|---|---|
| D decoding | HMM, $w = 0.2$, $L$ as measured | rule: min. stoppage / restart of 0, 1, 2, 4 s; HMM: $w \in \{1, 0.2, 0.05, 0.02\}$, $L \times \{0.5, 1, 2\}$ |
| H/A method | PathCRF + ball vote | PathCRF alone; ball rules (V5); ball-first per frame |
| ball distance | 2 m | 1 m, 2 m |
| min. possession $\tau$ | 2 s | 1, 2, 3, 4 s |

| setting (held-out DFL match) | spellF1 |
|---|---|
| clean tracking, full pipeline (rule for D) | 0.60 |
| degraded tracking, D model trained on clean data (rule for D) | 0.35 |
| degraded tracking, D model trained on degraded data + ball vote + $\tau = 2$ s, rule for D | 0.433 (0.42–0.45) |
| same, **HMM for D** (the submission) | **0.455 (0.43–0.47)** |
| degraded tracking, ball rules (V5) | 0.18–0.30 |

Ranges are over the three degradation seeds. The HMM beat the rule on every seed (+0.006, +0.039,
+0.021), and its neighbouring settings ($w = 0.05$ or $0.2$, any $L$ scale) all gained +0.019 to
+0.024: a plateau, not an isolated peak. It also predicts the number of stoppages much better
(91–95 vs. 92 true on the held-out match, alive/dead F1 0.93–0.94 vs. 0.92–0.93 for the rule).

The 0.455 is our only estimate of the expected score. It is measured on one held-out match, with a
simulated (not measured) degradation model, and the HMM setting was picked on that same match, so
treat it as a rough, slightly optimistic estimate.

## 5. Possession rules implemented

| situation | our rule |
|---|---|
| **Loose ball, passes, challenges** | PathCRF tracks the possession path. While the ball is between players, the frame belongs to the team of the last player in control, until the other team's player takes it. A change of team that lasts less than $\tau = 2$ s and is followed by the other team again reverts to it, so brief touches and contested balls do not split a possession. |
| **Ball evidence vs. PathCRF** | A whole possession is reassigned only when the ball contradicts it clearly (≥ 60% of ≥ 1 s of evidence frames, 2 m / 0.75 m margin). Single frames never override it. |
| **Dead ball** | `D` wherever the HMM decodes a stoppage (Section 2.3). There is no fixed minimum length: a stoppage is kept when the dead-ball evidence over its frames outweighs the cost of two state changes. |
| **Restarts** | After every `D`, possession starts fresh: the 2 s minimum is not applied across a stoppage. A brief restart between two stoppages stays `D` unless the evidence for live play outweighs the cost of the extra state changes. |
| **Half boundaries** | `X` outside the graded halves. Between the start of the second-half video block (frame 110808) and the second-half kick-off (frame 113742), `D`. Both boundaries are published in `DATA.md` / `FAQ.md`. |
| **Abstention** | We do not abstain (`X`) inside the halves. Every graded frame gets `H`, `A` or `D`. |

On BAR-ATH this gives 249 possessions with a mean of 11.9 s, Barcelona holding 61.9% of the
`H`/`A` frames, and `D` on 53.2% of graded frames. That share is slightly above the DFL range of
40–52% (Section 6).

## 6. Limitations, and what we would do with more time

- **The degradation is a model, not a measurement.** We matched fragment lengths, gaps, duplicates
  and ball loss to what we saw in the ATD, but not every failure mode. With more time we would fit
  the degradation parameters to the ATD statistics systematically and test several seeds.
- **Validation is on one held-out match.** We used three degradation seeds, but always the same
  test match. A leave-one-match-out loop would give a less optimistic estimate.
- **D is the weakest part.** On BAR-ATH we predict `D` on 53.2% of graded frames, slightly above
  every DFL match (40–52%). Either this match has more stoppage time, or the dead-ball model
  over-predicts on LaLiga tracking; without the ground truth we could not tell, and we did not
  tune on the competition match to find out. Next steps: train the D model on several degradation
  seeds, a semi-Markov model (explicit duration distributions instead of the HMM's geometric ones),
  and the smart-tagging set pieces as restart anchors.
- **PathCRF is used as-is.** It was trained on clean DFL tracking and Bundesliga football. Fine-tuning
  it on degraded, stitched DFL would likely help most.
- **Stitching errors propagate.** A forced join can swap two players. PathCRF is fairly robust to
  this, but it causes some wrong possession changes.
- **Ball evidence covers only 43% of live frames**, so most `H`/`A` decisions rest on PathCRF alone.

## 7. Reproduction

Full instructions are at the top of the repository's `README.md`. In short (Python 3.11):

```bash
pip install -r requirements-solution.txt     # NVIDIA GPU: install torch==2.15.0 from the cu132 index first
python src/run_all.py                        # add --cpu to force the CPU
```

Only `data/tactical/BAR-ATH_tactical_atd.txt` and `.xml` are read. The command writes
`Raumdeuters.csv`, validates it with the official validator, and prints its SHA256:

```
DDD20129A206F289A5BA62EE95686B617FF1E9319FAB24373A746AD5DD3821F8
```

We verified this hash in a fresh environment built only from `requirements-solution.txt`, both on an
NVIDIA GPU and on the CPU. A run takes about 1 minute with a GPU and about 5 minutes on the CPU.
To retrain the dead-ball model (optional, needs the DFL data):
`python src/train_dead_model.py --dfl <folder with the 7 preprocessed matches>`.

## References

- Kim H., Lee K., Seo S., Ko S.-K., Yoon J., Park C. *PathCRF: Ball-Free Soccer Event Detection via
  Possession Path Inference from Player Trajectories.* KDD 2026. arXiv:2602.12080,
  doi:10.1145/3770855.3818152. Code: https://github.com/hyunsungkim-ds/pathcrf (MPL-2.0).
- Bassek M., et al. *An integrated dataset of spatiotemporal and event data in elite soccer.*
  Scientific Data 12, 195 (2025). doi:10.1038/s41597-025-04505-y.
  Data: https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0).
- Pedregosa F., et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011.
