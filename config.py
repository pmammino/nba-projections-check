"""
Tunable parameters for the per-minute projection model.

EVERY NUMBER IN THIS FILE IS HAND-SET AND SHOULD BE TREATED AS A STARTING POINT.
None of it is fit to data yet. The `source` note on each block says where the
value came from and what would replace it.

Replace these by running calibrate.split_half_beta0() and
calibrate.fit_elasticities() once you have >=2 historical seasons of game logs
plus the preseason projections as they stood at the time.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stat vector definition
# ---------------------------------------------------------------------------
# Volume rates: modeled as counts per minute (Gamma-Poisson).
VOLUME_RATES = [
    "fga_pm",
    "fta_pm",
    "oreb_pm",
    "dreb_pm",
    "ast_pm",
    "stl_pm",
    "blk_pm",
    "tov_pm",
    "pf_pm",
]

# Conversion / share rates: modeled as successes per trial (Beta-Binomial).
# (numerator_col, denominator_col) in raw season-total or game-log space.
SHARE_RATES = {
    "pt3a_share": ("pt3a", "fga"),     # 3PA as a share of FGA -- role/style trait
    "fg2_pct": ("fg2m", "fg2a"),
    "fg3_pct": ("pt3m", "pt3a"),
    "ft_pct": ("ftm", "fta"),
}

ALL_STATS = VOLUME_RATES + list(SHARE_RATES)

# ---------------------------------------------------------------------------
# Prior strength (beta0)
# ---------------------------------------------------------------------------
# For VOLUME_RATES, units are MINUTES: beta0 = the number of minutes of observed
# play at which the preseason prior and the in-season data carry equal weight.
#
# For SHARE_RATES, units are TRIALS (attempts): a0 + b0 = pseudo-attempts.
#
# source: FITTED ON THREE INDEPENDENT SEASONS -- 2023-24, 2024-25 and 2025-26,
#   each with its own RotoWire preseason sheet. Per season: split at mid-January,
#   build the posterior from the preseason prior plus first-half games only,
#   score against each player's SECOND-HALF observed rate, grid search beta0 for
#   the minimum. The shipped values are the geometric mean across seasons.
#   See calibrate_multiseason.py.
#
# This objective beats split-half reliability because it optimizes forecast
# error directly rather than a proxy for it.
#
# IT REPLICATES. 12 of 13 land within 3x across the three seasons, and the
# blend beats a first-half-observed average in 38 of 39 stat-seasons and the
# preseason projection alone in 36 of 39. That is the strongest evidence in
# this project that a fitted block is a property of basketball rather than of
# one season.
#
# THE EARLIER SINGLE-SEASON FIT WAS OVERFIT IN PLACES. Shipping 2025-only
# values put stl_pm at 630 (three-season fit: 448) and blk_pm at 559 against
# per-season optima of 120 / 120 / 700. Measured cost of each blk_pm choice
# against each season's own optimum:
#
#   beta0      2023     2024     2025     mean
#    120      +0.00%   +0.00%   +5.76%   +1.92%
#    216      +0.66%   +1.13%   +2.72%   +1.50%   <- shipped
#    559      +7.78%   +8.78%   +0.01%   +5.52%   <- was shipped, 2025-only
#
# blk_pm is the one stat whose optimum does not transfer (5.8x spread), so 216
# is a deliberate compromise rather than a measurement. Everything else is.
#
# The hand-set values this replaced were too high almost everywhere. The
# reason the literature misled: stabilization estimates describe when an
# OBSERVED rate becomes reliable in isolation, not how much a PRESEASON
# PROJECTION should outweigh it, and the projection is a weaker anchor than
# that implies.
PRIOR_STRENGTH = {
    # --- per-minute counts, in minutes ---
    "fga_pm": 131,       # 2023/24/25 fits: 70 / 180 / 180
    "fta_pm": 203,       # 180 / 260 / 180
    "oreb_pm": 137,      # 120 / 120 / 180
    "dreb_pm": 198,      # 120 / 180 / 360
    "ast_pm": 157,       # 120 / 180 / 180
    "stl_pm": 448,       # 500 / 360 / 500
    "blk_pm": 216,       # 120 / 120 / 700  -- the one unstable stat, see below
    "tov_pm": 131,       # 180 / 180 / 70
    "pf_pm": 137,        # 180 / 120 / 120
    # --- conversion rates, in attempts ---
    "pt3a_share": 16,    # 20 / 10 / 20
    "fg2_pct": 84,       # 120 / 70 / 70
    "fg3_pct": 70,       # 70 / 70 / 70
    "ft_pct": 48,        # 70 / 40 / 40
}

# ---------------------------------------------------------------------------
# Recency decay (tau), in DAYS
# ---------------------------------------------------------------------------
# Evidence weight for a game d days ago is exp(-d / tau).
# Days rather than games, so an injury absence correctly ages the evidence.
#
# READ THIS BEFORE SHORTENING ANY OF THESE.
#
# tau and beta0 are in direct conflict, and the conflict is not obvious until
# you compute it. Exponential decay caps TOTAL evidence weight at an asymptote
# of roughly tau/2 games (at one game every two days). So the prior weight has
# a FLOOR it can never drop below:
#
#     pw_floor = beta0 / (beta0 + (tau/2) * mpg / phi)
#
# The first pass of this file used tau = 45-75 days. Run
# calibrate.implied_prior_floor() against those values and FGA/min never drops
# below 38% prior weight -- after a full 82-game season. That is wrong: by
# January you should largely trust observed FGA rate. Reaching a sensible floor
# (~15%) with beta0 = 320 requires tau around 160 days, which is the length of
# the regular season -- i.e. essentially NO within-season decay.
#
# That is the honest conclusion, and it is architectural rather than
# parametric: within one season you cannot have both a properly-strong prior
# and aggressive recency weighting. Pick one. This config picks the prior,
# because beta0 is the parameter with a principled estimator behind it
# (split-half reliability) while tau is a guess.
#
# The cost of a long tau is that genuine role changes -- a trade, a starter
# going down, a return from a long absence -- take too long to propagate. The
# right fix for that is a RESET, not faster global decay: zero out ex/em for
# the affected player, or shrink beta0, at the moment the role changes. See
# state.reset_player(). RotoWire already has the news feed that would trigger
# it, which makes this the cheap path.
#
# source: solved backwards from target prior-weight floors (see
#   calibrate.implied_prior_floor). Not fit to data.
DECAY_TAU_DAYS = {
    "fga_pm": 160,
    "fta_pm": 210,
    "oreb_pm": 215,
    "dreb_pm": 230,
    "ast_pm": 215,
    "stl_pm": 180,
    "blk_pm": 185,
    "tov_pm": 200,
    "pf_pm": 205,
    "pt3a_share": 170,
    "fg2_pct": 210,
    "fg3_pct": 250,
    "ft_pct": 260,
}

# ROLLING WINDOWS WERE TESTED TOO, AND ARE WORSE ON BOTH AXES. A hard
# last-N-games cutoff is a different weighting kernel from exponential decay
# (flat then zero rather than smooth), so it got its own test in
# compare_windows.py. Next-game rate MAE relative to full history, and
# game-to-game movement of the estimate:
#
#   scheme        full hist   W30    W20    W10     W5
#   rel MAE         1.000    1.0005 1.0005 1.0028  1.0084
#   rel jitter      1.000    1.13   1.39   2.04    2.74
#
# The jitter column is the important one and it inverts the intuition behind
# the question: full-history per-game updating is the MOST STABLE option,
# because one game added to 800 minutes of evidence barely moves anything. A
# five-game window moves its estimate 2.7x as much per game. Damping noise is
# what the prior and the accumulated evidence already do; a window removes
# that damping.
#
# Also tested: using a recent window only as a CONFLICT DETECTOR while keeping
# full history for the estimate (neutral on MAE, 0.996 on wrong priors but
# 1.067 worse on good ones -- the same trade as MIXTURE_PRIOR but weaker), and
# a two-window changepoint rule that discards stale evidence on divergence
# (1.0013 MAE, 1.62x jitter -- no gain, much noisier).
#
# Multiplier applied to every tau. Set below 1.0 to weight recency harder
# across the board -- and then re-run calibrate.implied_prior_floor() to see
# what it did to the prior floors, because it will raise all of them.
#
# LEAVE THIS AT 1.0. It was swept jointly with beta0 on the NEXT-GAME horizon
# (calibrate_decay.py), which is the horizon where recency should pay and the
# one the earlier beta0 tuning got wrong -- that scored against a whole
# second-half average, where the last week carries no special information by
# construction.
#
#   scale    0.04    0.08    0.15    0.30    0.50    1.00    2.00
#   rel MAE 1.0134  1.0061  1.0016  0.9998  0.9997  1.0000  1.0003
#
# Weighting recent games harder does not help the rate model, and weighting
# them much harder actively hurts: scale 0.04 (tau ~ 8 days) is 1.3% worse.
# The curve is flat from 0.30 to 2.00, so the shipped values sit inside the
# optimum. Best global setting gained 0.07% and per-stat retuning 0.10% --
# both noise.
#
# Tested conditionally too, since the obvious hypothesis is that recency
# matters for players whose ROLE just changed. It does not -- they do WORSE
# under decay than stable players (+2.8% vs +0.6% at scale 0.08). A role
# change moves a player's MINUTES, not his per-minute rates, so discarding
# evidence throws away a good estimate of a stable quantity.
#
# Recency belongs in the minutes model, and it is already there:
# MINUTES["tau_days"] = 10, a clean U-shaped optimum (tau 2 is 3.8% worse,
# tau 120 is 3.6% worse). Role changes fast; per-minute talent does not.
DECAY_SCALE = 1.0

# ---------------------------------------------------------------------------
# Overdispersion (phi)
# ---------------------------------------------------------------------------
# Real per-minute counts are overdispersed vs Poisson (matchups, foul trouble,
# role variance). We divide observed evidence by phi so the posterior does not
# become overconfident and does not update too fast.
# phi = 1.0 means "pure Poisson / Binomial".
#
# source: hand-set. Stats whose per-game value depends heavily on game context
#   get a higher phi.
# replace with: phi = Var(observed) / Var(Poisson-implied), by stat, from history
OVERDISPERSION = {
    "fga_pm": 1.35,
    "fta_pm": 1.55,
    "oreb_pm": 1.45,
    "dreb_pm": 1.40,
    "ast_pm": 1.60,
    "stl_pm": 1.30,
    "blk_pm": 1.40,
    "tov_pm": 1.45,
    "pf_pm": 1.30,
    "pt3a_share": 1.30,
    "fg2_pct": 1.25,
    "fg3_pct": 1.15,
    "ft_pct": 1.10,
}

# ---------------------------------------------------------------------------
# Robust prior: mixture component for prior-data conflict
# ---------------------------------------------------------------------------
# THE PROBLEM. The conjugate update is NON-ROBUST: its posterior mean is
# (alpha0 + X) / (beta0 + N), a fixed linear blend whose weight depends only on
# accumulated exposure, never on whether the data agrees with the prior. A
# player whose preseason projection is badly wrong gets shrunk exactly as hard
# as one whose projection is perfect, so the model is slow to abandon a prior
# it should abandon.
#
# THE FIX. Add a second, diffuse prior component centred on the population
# rate. The posterior weight between the two updates by marginal likelihood,
# in closed form (negative binomial for Gamma-Poisson, beta-binomial for
# Beta-Binomial). When the data contradicts the preseason projection the
# diffuse component takes over; when it agrees, the mixture is
# indistinguishable from the single-component posterior. Heavy-tailed priors
# give ADAPTIVE shrinkage -- this is the standard result and the standard fix.
#
# It behaves as designed. Measured weight on the diffuse component, by the
# standardized prior-data conflict of the observed-to-date rate:
#
#   |z|       <1      1-2     2-3     3-5      5+
#   weight   0.033   0.044   0.061   0.106   0.400
#
# WHAT IT IS AND IS NOT WORTH, and the two answers differ because they measure
# different things.
#
#   NEXT-GAME rate MAE:  0.07% better overall, 0.56% better in the highest
#   conflict bucket. Essentially nothing -- because next-game error is
#   dominated by irreducible single-game noise (the talent layer sits within
#   ~5% of the Poisson floor), so a better talent estimate barely moves it.
#
#   CONVERGENCE to the player's season-end rate -- the right metric for "slow
#   to update" -- normalized error, by games played:
#
#     prior genuinely wrong (|season gap| > 40%, 21.5% of rows, 56.6% of error)
#       games    1-3    4-6   7-10  11-20  21-40    40+
#       mixture 0.93   0.89   0.86   0.85   0.80   0.77   (vs conjugate = 1.00)
#
#     prior good (|season gap| <= 15%)
#       games    1-3    4-6   7-10  11-20  21-40    40+
#       mixture 1.09   1.08   1.05   1.02   1.00   0.99
#
#   So: 15% better where the prior is wrong, 2.5% worse where it is right, net
#   7.3% better. The cost lands on players whose error is already ~8x smaller,
#   so in absolute terms the trade is about 6:1 in favour.
#
# IT REPLICATES ACROSS THREE SEASONS, and the measured cost is SMALLER than the
# single-season figure above suggested. Convergence to the season-end rate,
# mixture relative to plain conjugate (below 1.0 is better):
#
#   season   prior WRONG   prior GOOD
#   2023-24     0.869        1.007
#   2024-25     0.907        1.010
#   2025-26     0.848        1.009
#   mean        0.875        1.009
#
# 12.5% better where the prior is genuinely wrong, 0.9% worse where it is
# right -- not the 2.5% measured on one season. Still off by default, because
# the reasoning has not changed: the gain is on CONVERGENCE, not on next-game
# MAE, so enabling it is a judgement about which you are optimizing. The
# three-season replication and the smaller cost both strengthen the case for
# turning it on for season-long, ROS and auction work.
#
# ON THE FIRST FEW GAMES SPECIFICALLY: the gain is SMALLEST there (7%) and
# grows with exposure. That is not a tuning failure, it is an information
# limit -- with two or three games you cannot separate a wrong prior from
# noise, and any scheme that reacts faster is chasing variance. Confirmed:
# uniformly weakening beta0 (the obvious alternative) was worse overall, and an
# explicit surprise-discount rule was worse than the mixture at every setting
# tried.
#
# ALSO WORTH KNOWING: conditioning on the observed-to-date rate shows an 8-19%
# apparent bias for under-performers. That looks alarming and is mostly NOT a
# flaw -- a correct shrinkage estimator conditioned on a noisy predictor shows
# exactly that pattern by construction. Fitting a per-cell bias correction on
# one half of the season and applying it to the other recovered 0.15% overall
# and made the first-three-games case worse. Do not chase it.
#
# source: compare_robust_priors.py, 76k scored player-games.
MIXTURE_PRIOR = {
    # Off by default: the gain is on the convergence metric, not on next-game
    # MAE, so turning it on is a judgement about which you are optimizing.
    # Season-long value, ROS projections and auction work want it on; a
    # single-slate DFS point projection barely notices.
    "enabled": False,
    # Prior probability that the preseason projection is the right component.
    # 0.90 says "usually well calibrated", which matches the data.
    "weight_preseason": 0.90,
    # Strength of the diffuse component as a fraction of beta0. Small means
    # "if the projection is wrong, fall back to nearly pure data".
    "diffuse_strength_frac": 0.05,
}


# ---------------------------------------------------------------------------
# Opponent profile layer
# ---------------------------------------------------------------------------
# A STAT-SPECIFIC opponent adjustment, separate from ELASTICITY's opponent
# coefficient and deliberately so. That one multiplies a single scalar per game
# (market-implied opponent efficiency) which cannot tell "allows blocks" from
# "allows assists"; nine of thirteen stats carried a zero there. This layer
# uses the opponent's own stat-allowed rate instead.
#
# source: FITTED out of sample -- Poisson with the posterior as offset so zeros
#   count, opponent profiles built walk-forward, train through 2026-02-11 and
#   scored on later games. 9,092 scored player-games. See fit_opponent_terms.py.
#
# Out-of-sample gain, market scalar vs stat profile, for the nine stats that
# previously had NO opponent term at all:
#
#   stat          market   profile   coef    shipped?
#   fta_pm        +1.18%   +1.91%    0.65    yes
#   fga_pm        +1.00%   +1.19%    0.81    yes
#   pt3a_share    +0.20%   +0.93%    0.93    yes
#   fg2_pct       +0.43%   +0.85%    1.05    yes
#   tov_pm        +0.39%   +0.77%    0.95    yes
#   ast_pm        -0.37%   +0.41%    0.92    yes
#   fg3_pct       +0.02%   +0.09%      --    no, noise
#   ft_pct        -0.20%   -0.14%      --    no, free throws are unguarded
#   oreb_pm       -0.55%   -0.56%      --    no, negative and unexplained
#
# Eight of nine prefer the profile over the market scalar, and the coefficients
# cluster near 1.0 -- the opponent's allowed rate transfers to the player
# roughly one-for-one, which is the clean reading.
#
# Three stats that ALREADY had a market opponent coefficient also earn a
# profile term on top: dreb (+0.54%, 0.89), blk (+0.57%, 0.81), pf (+1.41%,
# 0.74). Steals do not (+0.00%), so they keep their market term only.
#
# END-TO-END IT IS MUCH SMALLER THAN THE RATE-LEVEL FIT SUGGESTS, and the
# reason is worth knowing. On the full slate projection the layer is worth
# +0.03% of fantasy-point MAE -- not significant under a cluster bootstrap over
# games (95% CI -0.016 to +0.019). Turn TEAM RECONCILIATION OFF and the same
# comparison gives +0.36%, a twelve-fold difference, with every stat improving.
#
#   stat   reconcile on   reconcile off
#   FP        +0.03%         +0.36%
#   fga       -0.09%         +0.14%
#   ast       -0.05%         +0.30%
#   fta       +0.23%         +0.41%
#   tov       +0.81%         +0.62%
#   blk       +0.50%         +0.72%
#
# The possession constraint rescales team volume to the market-implied total,
# which undoes most of the opponent-specific level effect -- and that is
# CORRECT, because the market total is a better estimate of team volume than an
# opponent profile is. What survives is within-team redistribution plus the
# stats outside the reconciliation set (blocks, rebounds). Shipped because it
# is structurally right, costs nothing, and is opt-in via `opp_profiles`; not
# shipped on a claim that it moves the headline number.
#
# NOT UNIFORM ACROSS PLAYERS, and the pattern differs by stat type. For dense
# stats the gain is largest for LOW-rate players (FTA +14.0% bottom quartile
# vs +0.5% top); for sparse counts it is largest for HIGH-rate players and
# negative at the bottom (blocks -2.4% bottom, +1.6% third). Shipped uniformly
# because every included stat is net positive, but that heterogeneity is real
# and gating is a reasonable refinement.
OPPONENT_PROFILE = {
    "enabled": True,
    "beta": {
        # previously had no opponent term at all
        "fga_pm": 0.808,
        "fta_pm": 0.651,
        "ast_pm": 0.921,
        "tov_pm": 0.946,
        "pt3a_share": 0.932,
        "fg2_pct": 1.045,
        # already had a market term; the profile adds on top
        "dreb_pm": 0.893,
        "blk_pm": 0.813,
        "pf_pm": 0.742,
    },
    # Guard against an extreme profile from a thin sample reaching the rates.
    "mu_clip": (0.70, 1.40),
}


# ---------------------------------------------------------------------------
# Simulation dispersion
# ---------------------------------------------------------------------------
# SEPARATE FROM OVERDISPERSION ABOVE, DELIBERATELY, BECAUSE phi WAS DOING TWO
# UNRELATED JOBS WITH ONE NUMBER.
#
#   1. SHRINKAGE in the posterior: evidence enters as x/phi, so a larger phi
#      means slower updating. That role is redundant with PRIOR_STRENGTH --
#      both control how fast the posterior moves -- and since beta0 was fitted
#      empirically WITH phi held at the values above, phi's shrinkage
#      contribution is already baked into those fitted beta0 values. Changing
#      OVERDISPERSION now would silently invalidate them.
#
#   2. SAMPLING DISPERSION in the simulator: how much a single game's count
#      varies around its mean. That is a physical property of the stat and
#      should be measured, not inherited from a shrinkage knob.
#
# So job 2 gets its own table. Measured on 7,182 real backtest player-games as
# mean((x - lambda)^2 / lambda) with lambda the projected count:
#
#   stat      configured phi    measured
#   fga_pm         1.35           1.70
#   fta_pm         1.55           1.96
#   reb            ~1.40          1.56
#   ast_pm         1.60           1.33
#   stl_pm         1.30           1.15
#   blk_pm         1.40           1.08
#   tov_pm         1.45           1.05
#
# Wrong in BOTH directions -- the hand-set values were too low for the volume
# stats and too high for the defensive ones. Note these measurements conflate
# sampling noise with projection error, since lambda is a projection rather
# than the true rate, so they are upper bounds on pure sampling dispersion.
# Rate uncertainty contributes only ~0.3% of simulated band width, so the
# conflation is small, but it is a reason not to treat these as exact.
#
SIM_DISPERSION_MEASURED = {
    "fga_pm": 1.70,
    "fta_pm": 1.96,
    "ast_pm": 1.33,
    "stl_pm": 1.15,
    "blk_pm": 1.08,
    "tov_pm": 1.05,
    "oreb_pm": 1.56,
    "dreb_pm": 1.56,
}

# AND YET THE MEASURED VALUES ARE NOT THE DEFAULT. Substituting them made
# per-stat interval coverage WORSE, not better:
#
#   coverage   configured phi    measured phi    target
#   points         0.596            0.617         0.50
#   FGA            0.664            0.690         0.50
#   fantasy pts    0.522            0.521         0.50
#
# The reason is a double count. The measurement uses lambda = the PROJECTED
# count, so (x - lambda)^2 contains projection error as well as sampling
# noise. The simulator already draws rate uncertainty separately, so feeding
# it a dispersion that includes projection error inflates the band twice.
#
# Removing the projection-error component is the fix, and it is not done here.
# Until it is, SIM_DISPERSION stays empty and the simulator falls back to
# OVERDISPERSION, which calibrates better even though it was hand-set.
#
# The STRUCTURAL separation is kept regardless, because the two roles really
# are different numbers and conflating them is how this went unnoticed. Note
# this does NOT explain the per-stat over-coverage of points and FGA -- that
# remains open, and the measured phi pointed the wrong way on it.
SIM_DISPERSION: dict[str, float] = {}


def sim_dispersion(stat: str) -> float:
    """Sampling dispersion for the simulator, falling back to the shrinkage phi.

    No longer called by simulate.py / game_sim.py's own sampling step -- that
    step draws Binomial(minutes, rate) now (see the note below), which has no
    phi knob. Kept for calibrate.py and any other caller that still wants the
    old Poisson-mixture dispersion number.
    """
    return SIM_DISPERSION.get(stat, OVERDISPERSION.get(stat, 1.0))


# ---------------------------------------------------------------------------
# Simulator calibration: count model, spread, and tail shape
# ---------------------------------------------------------------------------
# Backtested against 18 held-out historical slates (2026-10-28 to 2026-11-15,
# ~3,158 player-games), scored as interval coverage: does the actual value
# land inside the simulated p25-p75 band as often as it should (target 0.50)
# and inside p05-p95 as often as it should (target 0.90)?
#
# 1. COUNT MODEL: Binomial(n=minutes, p=rate), not Poisson(rate*minutes).
#    Poisson has Var = lambda as a hard floor -- a per-stat dispersion sweep
#    (pushing the phi above down toward 0) could not narrow bands further
#    because phi < 1.0 collapses straight back to plain Poisson with zero
#    additional effect (see the SIM_DISPERSION_MEASURED note above for a
#    separate, earlier finding that substituting *measured* phi actually made
#    coverage worse). Binomial(n, p) has Var = n*p*(1-p) = lambda*(1-p),
#    strictly BELOW the Poisson floor whenever p > 0, and "minutes" is a
#    natural trial count -- you cannot record more assists than minutes
#    played, so this is a reinterpretation of the same mean, not a different
#    model pasted on top. It narrows bands wherever p isn't tiny (fga, fta,
#    and everything derived from them); it does very little for ast_pm,
#    stl_pm, blk_pm, which sit well under 0.2/min, where Binomial(n,p) is
#    numerically close to Poisson (see the known limitation below).
#
# 2. MINUTES SPREAD (SIM_MINUTES_SPREAD): the minutes residual bootstrap
#    (simulate._draw_minutes) is rescaled around its own median so its p25-p75
#    band hits the 0.50 target. Tuned jointly with (3) by grid search,
#    objective |coverage-0.5| averaged over minutes and fantasy points.
#
# 3. RATE SPREAD (SIM_RATE_SPREAD): the posterior Gamma/Beta rate draw's SD is
#    scaled by this factor, tuned jointly with (2).
#
# 4. TAIL SHAPE (SIM_TAIL_GAMMA): even with (2)/(3) tuned so cov_25_75 ~ 0.50,
#    cov_05_95 (the 90% interval) ran high -- 0.92 to 0.98 across every stat,
#    INCLUDING minutes and fantasy points where the center band was already
#    essentially exact. A single multiplicative spread can match the IQR or
#    the 90% band but not both if the true shape is more peaked (thinner
#    tails relative to its own IQR) than the empirical residual sample
#    implies, which is what this looks like. SIM_TAIL_GAMMA compresses only
#    the probability mass OUTSIDE the 25th/75th percentile toward the
#    quartile boundary (u=0.25 and u=0.75 are fixed points of the warp, so it
#    narrows p05/p95 without moving p25/p75 at all). Ruled out as the driver
#    by direct ablation: the shared per-game pace shock (delta ~0 on every
#    stat) and the 4.4% overtime event (delta ~0 on every stat). The leading
#    remaining explanation is that the empirical residual table's own extreme
#    quantiles, estimated from roughly 400-500 observations per minutes
#    bucket in this backtest, are themselves noisy -- not yet confirmed.
#
# KNOWN REMAINING LIMITATION. ast, stl, blk, tov, and reb still over-cover
# their 25-75 band (0.64 to 0.86 vs the 0.50 target) after all of the above.
# Directly ruled out: pure discreteness/atom artifact (restricting to rows
# where p25 != p75 still over-covers: ast 0.706, reb 0.643, pt3m 0.747,
# stl 0.853, blk 0.877), teammate/pace correlation (pace-shock ablation,
# delta ~0), and rate (Gamma) uncertainty -- a per-stat rate_spread sweep down
# to 5% of its tuned value moved these stats' coverage by at most 0.02,
# because their per-minute rates are small enough (p well under 0.2/min for
# ast_pm/stl_pm/blk_pm) that the Binomial "(1-p)" discount over Poisson barely
# applies there. What's left is the Binomial/Poisson sampling floor at the
# current mean rate -- real box scores for these specific stats may be
# genuinely under-dispersed relative to that floor (role/scheme consistency
# game to game), which would need a non-standard, sub-Poisson count model to
# fix. Not attempted here.
SIM_MINUTES_SPREAD = 0.94
SIM_RATE_SPREAD = 1.2
SIM_TAIL_GAMMA = 1.6


# ---------------------------------------------------------------------------
# Market environment elasticities
# ---------------------------------------------------------------------------
# Applied multiplicatively in log space:
#
#   log(rate_adj) = log(rate_posterior)
#                 + pace   * log(poss_hat / POSS_LEAGUE_AVG)
#                 + own    * log(own_ortg_hat / ORTG_LEAGUE_AVG)
#                 + opp    * log(opp_ortg_hat / ORTG_LEAGUE_AVG)
#                 + home   * is_home
#
# Sign logic worth reading before you touch these:
#   - oreb loads NEGATIVE on own efficiency  (your team missing = your O-boards)
#   - dreb loads NEGATIVE on opp efficiency  (opponent missing = your D-boards)
#   - blk/stl load POSITIVE on opponent volume, NEGATIVE on opponent efficiency
#   - ft_pct loads on nothing: free throws are unguarded
#   - pace elasticity < 1.0 for FGA because usage is sticky, not pace-proportional
#
# source: hand-set from the above reasoning, magnitudes guessed.
# replace with: calibrate.fit_elasticities() -- a per-stat log-link regression
#   with the player's posterior rate as an offset and partial pooling by position
ELASTICITY = {
    #              pace,  own_ortg, opp_ortg, home
    "fga_pm":     (0.85,   0.15,    0.00,     0.000),
    "fta_pm":     (0.70,   0.25,    0.00,     0.010),
    "oreb_pm":    (0.90,  -0.30,    0.00,     0.000),
    "dreb_pm":    (0.90,   0.00,   -0.35,     0.000),
    "ast_pm":     (0.90,   0.40,    0.00,     0.005),
    "stl_pm":     (0.95,   0.00,   -0.25,     0.000),
    "blk_pm":     (0.85,   0.00,   -0.20,     0.000),
    "tov_pm":     (0.85,  -0.15,    0.00,     0.000),
    "pf_pm":      (0.60,   0.00,    0.10,    -0.020),
    "pt3a_share": (0.00,   0.05,    0.00,     0.000),
    "fg2_pct":    (0.00,   0.35,    0.00,     0.004),
    "fg3_pct":    (0.00,   0.25,    0.00,     0.004),
    "ft_pct":     (0.00,   0.00,    0.00,     0.000),
}

# League-average anchors for the elasticity terms.
# source: MEASURED from 231 real 2025-26 games (ESPN team box scores), via
#   POSS = FGA - OREB + TOV + 0.44*FTA. The first pass of this file hand-set
#   POSS_LEAGUE_AVG = 99.0, which is 3.4% low -- that error biased every
#   pace_mult by log(102.5/99) = +0.035, i.e. a systematic ~3% overestimate on
#   every pace-elastic rate. Refresh both each season.
# Re-measured on the full season (978 games) rather than the 231-game sample,
# and confirmed stable out of season: 2024-25 gives 101.0 and 112.6, a drift of
# -0.6% and -0.9%. These are real league constants, not season artifacts.
POSS_LEAGUE_AVG = 101.5   # team possessions per 48 min (was 102.5 on 231 games)
ORTG_LEAGUE_AVG = 113.6   # points per 100 possessions

# ---------------------------------------------------------------------------
# Possession model
# ---------------------------------------------------------------------------
# poss_hat = intercept + b_pace * team_pace_prior + b_total * market_total
#
# THIS REPLACES AN ASSUMPTION THAT TURNED OUT TO BE WRONG.
#
# The original design said: the market total cannot identify pace (total =
# pace x efficiency), so anchor pace on a team pace prior and attribute the
# total's residual to efficiency. Tested against 175 real 2025-26 games with
# DraftKings closing lines, that is backwards:
#
#   corr(total, actual possessions)     = +0.383
#   corr(pace prior, actual possessions)= +0.340
#   corr(total, actual per-team ORtg)   = +0.210
#   corr(pace prior, actual ORtg)       = -0.002
#
# The total predicts pace BETTER than a team pace prior does, and predicts
# efficiency only weakly. So the total belongs in the possession estimate, not
# purely in the efficiency channel. Both carry partly independent information
# (R2 0.115 prior-only, 0.146 total-only, 0.172 together), which is why the
# model below uses both.
#
# Effect: possession MAE drops from 4.73 (league constant 99) to 3.65.
#
# source: OLS on 175 games. THAT IS A SMALL, SAMPLED SLICE (~14% of a season)
#   and these coefficients are noisy. Refit on a full season with
#   calibrate.fit_possession_model().
# REFIT ON 978 GAMES, UP FROM 175. The first version used the 231-game ESPN
# team-box sample and was badly under-identified -- refitting the SAME season
# on the full set of games with closing lines moves the coefficients enormously
# (intercept -6.42 -> -79.5, pace 0.651 -> 1.401). That is sample instability,
# not season drift, and it was only visible once the out-of-season work built a
# full-season team box by summing player logs.
#
# Held-out check, fit on 586 games and scored on 392 later ones:
#
#   possession MAE    shipped coefficients  3.596
#                     refit coefficients    3.450   (-4.1%)
#                     pace prior alone      3.826
#                     league constant       4.348
#
# So the refit is measurably better, and R2 roughly doubles (0.17 -> 0.31).
# Note the intercept is large and negative with a pace coefficient above 1;
# that is a fine extrapolation-free fit inside the observed range of pace and
# total but should not be trusted outside it, which is why poss_hat is clipped
# in market.game_environment.
POSS_MODEL = {
    "intercept": -79.509,
    "b_pace_prior": 1.401,
    "b_total": 0.166,
}

# Fallback when no team pace prior is available (early season, new team):
# drop the pace_prior term and re-solve the intercept so the mean is preserved.
POSS_MODEL_NO_PRIOR = {
    "intercept": 25.860,
    "b_pace_prior": 0.0,
    "b_total": 0.328,
}

# ---------------------------------------------------------------------------
# Minutes model
# ---------------------------------------------------------------------------
# source: prior_games and tau_days are MEASURED (see below); the blowout curve
#   and back-to-back multipliers are still hand-set. The blowout curve is the
#   piece most worth fitting next -- regress actual minutes on |spread| by
#   depth role.
MINUTES = {
    # Prior strength for preseason min/game, in GAMES.
    #
    # ZERO, AND THAT IS A MEASURED RESULT, NOT AN OVERSIGHT.
    #
    # This was hand-set to 8.0. The validation suite then found that an arm
    # ignoring the preseason minutes projection entirely beat the shipped
    # blend by 0.67 minutes of MAE and 0.34 fantasy points, both significant
    # under a cluster bootstrap over slates. A direct sweep confirmed it:
    # minutes MAE over 40 walk-forward slates, by prior_games (tau = 10 days)
    #
    #   prior_games   0.0     0.5     1.0     2.0     4.0     8.0    16.0
    #   minutes MAE  4.765   4.820   4.885   5.037   5.293   5.611   5.900
    #
    # Monotone, at every tau tested. Going from the shipped 8.0 to 0.0
    # improves minutes MAE by 12.8%.
    #
    # Why: the sheet's minutes/games is a full-season average of a role as
    # forecast in October, and the sheet over-allocates team minutes by ~9% on
    # top of that. Observed minutes from the last couple of weeks describe the
    # rotation the coach is actually using. This is specific to MINUTES -- the
    # preseason projection remains valuable for per-minute RATES, where the
    # posterior beats observed-only on 12 of 13 stats.
    #
    # 0.0 does NOT mean the preseason projection is unused. baseline_minutes
    # falls back to it whenever a player has no observed minutes at all, which
    # is every player on opening night.
    "prior_games": 0.0,
    # Half-life for recent minutes, in days. MEASURED, and a clean U-shape
    # with the optimum exactly here:
    #
    #   tau    2      3      5      7     10     14     21     45    120
    #   MAE  5.128  5.040  4.970  4.944  4.942  4.947  4.977  5.044  5.121
    #
    # Both directions cost ~4%. This is where recency weighting earns its
    # keep -- and the contrast with the RATE decay above is the whole point:
    # rates want a season-long memory, minutes want about ten days.
    "tau_days": 10.0,
    # source: hand-set, not yet fitted. Caps how much a single inter-game gap
    # can age a game's decay weight (minutes.baseline_minutes), so a long
    # injury absence does not make the pre-absence games look stale relative
    # to the comeback game the instant a player returns -- see the comment
    # there for the real case (Jerami Grant) this fixes. 7 days covers a
    # normal rest cadence, a standard back-to-back-avoidance gap, and most of
    # an All-Star break without also swallowing a genuine short-term injury.
    # Worth fitting properly (regress against how fast a RETURNING player's
    # minutes actually stabilize) rather than left at this starting guess.
    "gap_cap_days": 7.0,
    # Blowout / garbage time. Exposure begins once |spread| exceeds the
    # threshold, ramps linearly, and is capped.
    "blowout_threshold": 8.0,
    "blowout_scale": 10.0,    # points of spread per 1.0 unit of exposure
    "blowout_cap": 1.4,
    # Per-role response to one unit of blowout exposure (multiplicative).
    # Negative = loses minutes in blowouts. Renormalization to 240 then
    # redistributes what the starters give up.
    "blowout_role_beta": {
        "starter": -0.105,     # proj >= 28 mpg
        "rotation": -0.015,    # 15-28 mpg
        "bench": 0.130,        # < 15 mpg
    },
    "role_cut_starter": 28.0,
    "role_cut_rotation": 15.0,
    # Back-to-back: applied to the second night.
    #
    # MEASURED, and smaller than these imply. 18.0% of player-games are the
    # second night of a back-to-back. Minutes minus the player's own mean,
    # back-to-back versus not:
    #
    #   starter  -0.32 min      rotation  +0.77 min      bench  -0.48 min
    #
    # Inconsistently signed and all under one minute. The shipped 0.975 implies
    # about -0.8 min for a 32-minute starter, so it is directionally right and
    # roughly 2.5x too strong; the bench multiplier has the wrong sign. Softened
    # toward the measurement rather than removed, because the starter direction
    # is real and the rotation/bench numbers are small enough to be noise.
    #
    # Note this was only measurable once the matchup work computed a
    # back-to-back flag -- the layer ablation showed exactly 0.00% because no
    # backtest ever passed `b2b_teams`.
    "b2b_starter_mult": 0.990,
    "b2b_bench_mult": 1.000,
    # Team total minutes per regulation game.
    "team_minutes": 240.0,
    # Hard floor/ceiling on any single player's projected minutes.
    "min_floor": 0.0,
    "min_ceiling": 44.0,
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
# Below this many projected minutes a player is not projected at all.
MIN_MINUTES_TO_PROJECT = 1.0

# Cold start: players with no team in the preseason sheet (unsigned / two-way)
# get their prior strength multiplied by this, since the projection itself is
# more speculative.
NO_TEAM_PRIOR_MULT = 0.6


# ---------------------------------------------------------------------------
# Absence response
# ---------------------------------------------------------------------------
# How the remaining players change when rotation players are ruled out.
#
# Predictor is `absent_usage_share` in [0, 1]: the fraction of the team's
# rotation usage load (minutes x possession-events per minute) that is
# unavailable. One star out is typically 0.10-0.18.
#
# `beta` is applied as rate * exp(beta * tier_mult * pos_mult * absent_share),
# matching the market-environment form so the two compose without either
# knowing about the other.
#
# source: MEASURED on all 1,235 games of 2025-26 (32,261 player-games) from
#   ESPN player box scores plus DNP flags. Poisson-style weighted fits for
#   counts with the leave-one-out baseline rate as an offset; attempt-weighted
#   OLS on the log ratio for conversion rates. These are real fits, unlike
#   PRIOR_STRENGTH and ELASTICITY above. See fit_absence.py.
#
# T-STATISTICS MATTER HERE, so they are recorded per line. The headline result
# is that VOLUME and SHOT MIX respond strongly and reliably while SHOOTING
# EFFICIENCY DOES NOT RESPOND MEASURABLY AT ALL:
#
#   fga_pm      +15.3    fg2_pct    -1.3
#   fta_pm      +11.3    fg3_pct    -0.6
#   tov_pm       +8.5    ft_pct     +0.6
#   pt3a_share   -7.0    stl_pm     -0.4
#   ast_pm       +6.5    blk_pm     +1.4
#   pf_pm        -4.7    oreb_pm    -1.7
#                        dreb_pm    +2.0
#
# So when the primary creator sits, the others do NOT become worse shooters.
# They shoot more, get to the line more, turn it over more, and take FEWER
# THREES -- the assisted catch-and-shoot threes disappear, so the shot mix
# moves inside. The scoring-efficiency loss is a SHOT MIX effect (t = -7.0),
# not a shooting effect (t = -1.3). The fg2_pct/fg3_pct/ft_pct betas below are
# retained because their direction is theory-backed and their magnitude is
# small either way, but they are NOT statistically distinguishable from zero
# even with a full season. Set `apply_shooting_betas = False` to zero them.
#
# Pooled TS% response to absent usage is also not significant: -0.0321 with
# t = -1.6 (naive) or -1.3 (aggregated to team-game). Reported honestly rather
# than dressed up.
# ARCHETYPES WERE TESTED AND REJECTED. K-means on standardized role features
# (usage, 3PA share, FTA rate, assist/rebound/block/steal rates) at k = 4..10,
# plus a continuous Gaussian role-similarity kernel, all fit on a training fold
# and scored on held-out games. None beat the position parameterization on any
# channel, and the similarity kernel was WORSE than a pooled single predictor.
# The reason is visible in the clusters: they recover position plus a usage
# axis (one cluster is 81% C, another is 63% G with high assist rate), and the
# usage axis is already in the model as `tier_mult`. See compare_archetypes.py.
ABSENCE = {
    "beta": {
        # strongly identified
        "fga_pm": 0.197,        # t = +15.3
        "fta_pm": 0.393,        # t = +11.3
        "tov_pm": 0.250,        # t =  +8.5
        "pt3a_share": -0.149,   # t =  -7.0  creation loss -> shot mix inside
        "ast_pm": 0.160,        # t =  +6.5
        "pf_pm": -0.124,        # t =  -4.7
        # marginal
        "dreb_pm": 0.046,       # t =  +2.0
        "oreb_pm": -0.054,      # t =  -1.7
        # not significant; retained small, direction is theory-backed
        "blk_pm": 0.045,        # t =  +1.4
        "fg2_pct": -0.026,      # t =  -1.3
        "stl_pm": -0.012,       # t =  -0.4
        "fg3_pct": -0.017,      # t =  -0.6
        "ft_pct": 0.013,        # t =  +0.6
    },
    # Zero the three shooting-percentage betas, which are not significant.
    "apply_shooting_betas": True,
    "shooting_betas": ("fg2_pct", "fg3_pct", "ft_pct"),

    # --- minutes, additive, in minutes per 1.0 of absent share ------------
    # Parameterized by the ABSENT player's position, plus a bonus when the
    # survivor shares that position.
    #
    # This beat five alternatives out of sample on held-out games (train
    # through 2026-01-25, test on the remaining 547). oos R2 on minutes:
    #
    #   this spec (absent-pos + same-pos bonus by survivor pos)   0.0402
    #   absent-pos + single same-pos bonus                        0.0362
    #   absent-pos only                                           0.0349
    #   same/cross position (the previous parameterization)       0.0313
    #   same/cross ARCHETYPE (k-means on role stats)              0.0303
    #   full 3x3 absent-pos x survivor-pos interaction            0.0305  (overfits)
    #   pooled absent share                                       0.0296
    #   continuous role-similarity kernel                         0.0216  (worse than pooled)
    #
    # source: MEASURED, see compare_archetypes.py.
    "beta_minutes_by_absent_pos": {"C": 9.339, "F": 14.184, "G": 13.104},
    # Extra minutes for survivors who share the absent player's position.
    # Centres are the outlier and it is not subtle: +16.8 against +4.5 and
    # +5.2. A team carries two or three centres, so one out must be covered by
    # another centre; guards and forwards are interchangeable enough that the
    # bonus is small.
    "beta_minutes_same_pos_bonus": {"C": 16.814, "F": 4.512, "G": 5.239},
    "beta_minutes_default": 12.18,
    "beta_minutes_same_bonus_default": 5.0,

    # --- position of the ABSENT player scales the RATE response ------------
    # Usage response per 1.0 of absent share, by the absent player's position:
    #   C +0.096 (t=4.0)   F +0.119 (t=7.5)   G +0.139 (t=8.7)
    # against a pooled +0.110. Expressed as multipliers on the pooled per-stat
    # betas above.
    #
    # Guards vacate the most usage per unit of absent share. A SAME-position
    # term on usage was NOT significant (-0.005, t=-0.3) and the simpler
    # absent-position-only spec won out of sample, so unlike minutes the usage
    # response depends only on WHO LEFT, not on whether they played your spot.
    "usage_mult_by_absent_pos": {"C": 0.873, "F": 1.082, "G": 1.264},
    "usage_mult_default": 1.0,

    # --- efficiency: guard absence is the only one that bites --------------
    # TS% response per 1.0 of absent share, by the absent player's position:
    #   C +0.011 (t=+0.2)   F -0.037 (t=-0.8)   G -0.123 (t=-2.6)
    #
    # THIS RESOLVES THE EARLIER NULL RESULT. Pooling every absence together
    # gave an insignificant TS% effect (-0.032, t=-1.6) and the conclusion
    # "efficiency does not respond". Separating by the absent player's position
    # shows it does -- but only for GUARDS, which is the creation-loss channel
    # showing up where theory says it should.
    #
    # Multipliers on the pooled shooting betas, relative to the pooled -0.032.
    # SHRUNK 50% because no specification improved out-of-sample R2 on TS% --
    # every one of the nine tested was negative. So the coefficient is
    # significant in sample and theory-signed, but does not yet demonstrably
    # improve prediction. Shrinkage is the honest middle.
    "shooting_mult_by_absent_pos": {"C": -0.17, "F": 0.58, "G": 1.92},
    "shooting_mult_default": 1.0,
    "shooting_shrinkage": 0.5,

    # --- role tiers on the RATE response ----------------------------------
    # Low-usage players absorb the least extra creation per unit of absence
    # (+0.006 events/min per 10pp vs +0.014 for mid-high) and pay the most
    # efficiency for it (IV slope -1.07 vs -0.07). A uniform bump inverts this.
    "tier_mult": {
        "low": 0.45,
        "mid-low": 0.85,
        "mid-high": 1.10,
        "high": 1.05,
    },
    # Tier multipliers on the MINUTES response run the OTHER WAY: bench players
    # gain most (+1.46 min per 10pp vs +0.85 for high usage) because stars are
    # already near their ceiling.
    "tier_mult_minutes": {
        "low": 1.20,
        "mid-low": 1.10,
        "mid-high": 0.83,
        "high": 0.70,
    },
    # Fixed usage-per-minute tier cut points: mid-low / mid-high / high. Fixed
    # rather than within-team quantiles, because a quartile split on a 10-man
    # roster would label a team's third option "high usage".
    # source: measured quartile boundaries of base usage/min, 2025-26.
    "usage_tier_cuts": (0.383, 0.456, 0.549),

    # Rotation threshold (mpg) for the absent-share denominator.
    "rotation_min_mpg": 12.0,

    # --- the usage-efficiency tradeoff, for reference ----------------------
    # OLS on game-level usage variation gives +0.0049 pts/TSA per event/min.
    # THE SIGN IS WRONG, because within a game usage and efficiency are
    # positively selected: a player who is shooting well keeps getting the
    # ball. Absence is exogenous to that and identifies the causal slope by
    # instrumental variables: -0.291. Any model that fits this tradeoff on
    # game-level usage without an instrument gets the direction backwards.
    "usage_efficiency_slope_iv": -0.291,
    "usage_efficiency_slope_ols_biased": 0.0049,
}
