# Rat-a-Tat Cat — 2-player engine and solver

A rules-exact engine for the card game Rat-a-Tat Cat, a belief model over the
hidden cards, and sampling search agents that play it near-optimally. Pure
Python, no dependencies (pytest only for the tests).

```bash
python -m ratatatcat play --hint            # play a round against the solver
python -m ratatatcat advise --hand "3 ? ? 7" --discard "5 8 2" --drawn 4
python -m ratatatcat bench --a ismcts:2000 --b heuristic --rounds 200
python -m ratatatcat selfplay --a ismcts:1000 --b heuristic
```

## Headline finding

**The search only reaches parity with a good heuristic, and that is the honest
result.** The shipped ISMCTS config wins ~54% of rounds but its mean score
margin is +0.00 ± 0.58 points — indistinguishable from zero. Read the
[results](#measured-results) before assuming the solver is strong. The
interesting content of this project is *why* a well-calibrated heuristic is hard
to beat here, and three measured biases that made naive search actively worse
than doing nothing.

## Why "solve" means sampling here

Exact solution is off the table, and it is worth being precise about why. The
round is an imperfect-information game: each player knows two of their four
cards at the start and neither ever sees the opponent's hand. That partitions
the tree into information sets, so there is no single tree to solve with
minimax. Every deck draw is a chance node on top. Computing an exact Nash
equilibrium (CFR over the full tree) is not tractable without an abstraction so
coarse it would answer a different game.

What is tractable is **sampling**: repeatedly guess a full deal consistent with
what you know, search that, aggregate. "Near-optimal" below means *approximately
optimal under sampling*, never proven-optimal.

## The agents

| agent | what it does |
|---|---|
| `random` | uniform over legal moves; a floor |
| `heuristic` | rule-based, belief-aware; strong, and the rollout policy for both searchers |
| `pimc` | Perfect Information Monte Carlo — sample a world, solve it as perfect information, average |
| `ismcts` | Information Set MCTS — one tree of information sets across many sampled worlds |

Spec syntax: `ismcts:4000` sets iterations, `pimc:80x3` sets worlds × depth.

### Measured results

Seats alternate so first-player advantage cancels, and both agents see identical
deals. Margin is in points, positive favouring the first-named agent; `±` is one
standard error. Reproduce any row with `python -m ratatatcat bench`.

| match | rounds | win rate | mean margin |
|---|---|---|---|
| `heuristic` vs `random` | 600 | 77.7% ± 1.7% | +6.46 ± 0.34 |
| `ismcts:2000` vs `random` | 200 | 74.2% ± 3.1% | +5.91 ± 0.59 |
| `ismcts:1500` vs `heuristic` *(shipped config)* | 160 | 54.4% ± 3.9% | **+0.00 ± 0.58** |
| `ismcts:2000` vs `heuristic`, `exploration=0.25` | 300 | 54.3% ± 2.9% | +0.27 ± 0.45 |
| `ismcts:1500` vs `heuristic`, `exploration=1.2` | 160 | 50.6% ± 4.0% | −0.11 ± 0.58 |
| `ismcts:1500` vs `heuristic`, `backup="mean"` | 200 | 47.5% ± 3.5% | −0.54 ± 0.56 |
| `ismcts:1500` vs `heuristic`, `exploration=0.25` | 160 | 46.9% ± 3.9% | −1.02 ± 0.64 |
| `ismcts:3000` vs `heuristic`, naive SO-ISMCTS | 200 | 46.5% ± 3.5% | −1.18 ± 0.50 |
| `pimc:60x3` vs `heuristic` | 200 | 46.8% ± 3.5% | −1.50 ± 0.55 |

Read that table carefully. The win rates order the configurations sensibly and
the fixes move things in the right direction, but **the top rows are separated by
less than their combined standard errors**. The defensible claims are that the
heuristic clearly beats random, that naive SO-ISMCTS and PIMC clearly *lose* to
the heuristic, and that the tuned search recovers to roughly parity. "The solver
beats the heuristic" is not established by this data.

The `vs random` pair is the sharpest evidence: the search wins 74.2% ± 3.1%
where the heuristic alone wins 77.7% ± 1.7%. Even against a player making
uniformly random moves, thousands of simulations per decision buy nothing over
the rule they are simulating.

The heuristic beats random decisively, so the game is not trivial and the
measurement harness works. The search then fails to separate from the heuristic.
Card play is at parity — unsurprising, since correct play is close to "replace
your worst slot whenever the new card beats it", and the heuristic already does
exactly that with correctly computed expectations. There is little room above it.

## Three things that made naive search worse

Each was found by measurement, not inspection, and each is switchable so the
comparison stays reproducible.

**1. A stale player index flipped the sign of the value estimate.** Tree nodes
originally cached which player was to move at node-creation time. With the
opponent played as part of the environment, an arbitrary number of their moves
happen between two of our nodes, so that cached value is stale by the time the
node is used for selection — and the search ended up *minimising* its own value
at alternating depths. The mover is now read from the live state during descent.

**2. Mean backup made the agent call far too often.** This one is the most
interesting. Scoring 150 knock decisions against ground truth (fix the real
hands, reshuffle only the unseen deck, force each action, play both seats with
the heuristic, average) showed mean backup estimating the value of *calling* to
within 0.1 points, while undervaluing *passing* by 2.8. The asymmetry is
structural: calling ends the round almost immediately, so its average is a clean
estimate, whereas passing opens a subtree full of our own future decisions whose
average is dragged down by exploring bad branches. Comparing them is apples to
oranges.

Scoring by best line instead (`backup="pv"`) overshoots the other way — a
maximum over noisy estimates is optimistic, measured at +3.6 on passing. The
default `backup="mix"` blends the two at `pv_weight=0.6`, which measured best
end-to-end. Reproduce with:

```bash
python -m analysis.knock_calibration --sweep
```

For scale, on those 150 positions *never calling* gives away 237.5 points
against the clairvoyant oracle and *always calling* gives away 355. The best
simple rule found gives away 235.7 — barely better than never calling. The knock
decision carries about 4 points of swing on average but almost none of it is
predictable from what a player can actually see, which is a large part of why
there is so little headroom above the heuristic.

**3. Exploration lock-in at low iteration counts.** The exploration constant was
first set to 0.25, reasoning that normalised rewards only span ~0.28 standard
deviations so a textbook 1.4 would drown the signal. That starved exploration
instead. In one position holding `[8, ?, ?, 6]` with a 0 on the discard, an
800-iteration search gave the correct move (replace the known 8) just 58 visits
against 689 for a worse one, and never recovered; at 4000 iterations it ranks
them correctly. Of `{0.25, 0.6, 1.2}` the middle value measured best, though the
match differences were inside noise — the decisive evidence is the lock-in case,
not the win rate. Hence `exploration=0.6`, and `advise` defaulting to
`ismcts:4000`.

**PIMC's over-calling is a different bug, and inherent.** It called 171 times in
200 rounds against the heuristic's 29 — textbook **strategy fusion**. Inside
each sampled world PIMC can see its own face-down cards, so it "knows" whether
it is ahead. At the table it cannot tell those worlds apart, so it acts on
information it does not have. This is not fixable within PIMC; it is kept as an
honest baseline that makes the pathology visible.

## Rules, and where the sources disagree

Deck (54 cards): numbers 0–8 four times each, **nine** 9s, three each of Peek,
Swap and Draw 2.

Each player gets four cards face down and may look at the outer two. On your
turn you either take the top discard (replacing one of your cards) or draw from
the deck, then keep or discard it. Power cards drawn from the deck may be played
for their effect. You may call "Rat-a-Tat Cat" at the end of your turn; the
opponent gets one final turn, then hands are revealed and the lower total wins.

Published sources conflict on four points. Each is a flag in
[`rules.py`](ratatatcat/rules.py) rather than a buried assumption:

| flag | default | why |
|---|---|---|
| `deal_power_cards` | `False` | Sources say a power card dealt into your hand has no power and must be replaced by the first number card you draw. Rather than model forced replacement, the default deals number cards only. Keeps every hand slot in 0–9, which is what makes the belief model tractable. |
| `draw2_chains` | `False` | UltraBoardGames says a Draw 2 drawn during a Draw 2 chains; other summaries say power cards drawn that way are inert. Inert is the default because it always terminates. |
| `knocker_protected` | `False` | The rulebook does not say whether the caller can be hit by a Swap on the final turn. Default follows the "normal play continues" reading. |
| `power_card_score` | `9` | Only reachable with `deal_power_cards=True`; no source states a value. |

One rule the sources agree on and the engine enforces: **a power card in the
discard pile is spent** and cannot be taken. Only number cards may be drawn from
the discard.

Sources: [Wikipedia](https://en.wikipedia.org/wiki/Rat-a-Tat_Cat),
[UltraBoardGames](https://www.ultraboardgames.com/rat-a-tat-cat/game-rules.php),
[GroupGames101](https://groupgames101.com/rat-a-tat-cat-rules/),
[Gamewright rulebook (PDF)](https://gamewright.com/pdfs/Rules/Rat-a-TatCat-RULES.pdf).

## How the belief model works

[`belief.py`](ratatatcat/belief.py) does two things.

**Card counting.** Everything you have observed — your own known slots, every
card ever discarded, opponent cards revealed when they take from the discard —
is subtracted from the 54-card deck. What is left is the pool every hidden card
must come from.

**Provenance.** Uniform-over-unseen is only right for a slot nobody has acted
on. The engine tracks how the card in each slot got there, which is genuinely
public — at a table you see *which* slot someone replaces, peeks at or swaps,
even when the card's face stays hidden:

- `kept_from_deck` / `took_discard` — the owner saw it and chose it, so it skews low
- `dealt_seen` — the owner has looked and left it, so it skews low, mildly
- `dealt` — never seen by anyone, so it skews slightly *high*: a player who knew
  it was low would have kept it, one who knew it was high would have replaced it

Set `use_provenance=False` on either search agent to isolate the effect. Note it
is **not** what caused the over-calling — measured, the bias toward calling was
+2.73 with the priors on and +2.71 with them off.

**Determinization** combines the two into a concrete legal world consistent with
one player's information. Correctness here is what keeps the search honest, and
it is enforced by tests: a determinized world always contains all 54 cards,
never contradicts a card the player knows, and never echoes the true hidden hand.
Agents read only `state.knowledge[me]`, never `state.hands` —
`test_agents_do_not_read_the_hidden_hand` scrambles the cards an agent cannot see
and asserts its decision does not change.

## Analysing a real position

```bash
python -m ratatatcat advise --hand "3 ? ? 7" --discard "5 8 2" --drawn 4
```

```
Position:
   you: [ 3  ?  ?  7]
    P1: [ ?  ?  ?  ?]
  discard top: 2   deck: 42
  you drew: 4

  move                        samples   value
  keep drawn card -> slot 3      1430    0.652
  keep drawn card -> slot 1       382    0.634
  keep drawn card -> slot 2       159    0.614
  keep drawn card -> slot 0        15    0.475
  discard drawn card               14    0.461

  recommended: keep drawn card -> slot 3
```

`--hand` takes four slots, `?` for cards you cannot see. `--opp` gives any
opponent cards you know. `--discard` is the pile oldest-first, which feeds the
card counting — the more of it you supply, the sharper the read. Impossible
positions (a fifth 3, say) are rejected rather than silently fudged.

`value` is on a normalised [0, 1] scale; `(2 * value - 1) * REWARD_SCALE`
converts it back to a score margin in points.

## Layout

```
ratatatcat/
  cards.py       deck composition, parsing
  rules.py       RuleConfig — every contested rule is a flag here
  state.py       game state, legal moves, transitions, knowledge + provenance
  belief.py      card counting, priors, determinization
  advisor.py     build a state from a table-side description
  simulate.py    head-to-head matches with standard errors
  cli.py         play / advise / bench / selfplay
  agents/
    base.py      Agent protocol, random baselines
    heuristic.py rule-based player, also the rollout policy
    search.py    ISMCTS and PIMC
analysis/
  knock_calibration.py   ground-truth scoring behind the tuning constants
tests/           140 tests
```

```bash
python -m pytest -m "not slow"   # fast suite, ~1s
python -m pytest                 # includes the statistical benchmarks
```

## Known limitations

- **The search's edge is not statistically established.** +0.27 ± 0.45 over 300
  rounds. Confirming a real effect of that size needs a few thousand rounds.
- **It still over-calls.** 211 calls to the heuristic's 89 over 300 rounds, even
  after the backup fix. Ground truth suggests the right rate is nearer 45%.
- **Single-observer ISMCTS.** The tree models our information sets, not the
  opponent's. Multi-observer ISMCTS would model their uncertainty about *us* —
  relevant for bluffing with an early call.
- **The opponent model is fixed.** `model_opponent="heuristic"` cannot adapt to
  an opponent who plays differently. `model_opponent="search"` restores textbook
  SO-ISMCTS and is measurably worse. A self-play loop that periodically replaces
  the model with the current best agent is the natural next step.
- **The provenance priors are hand-tuned**, not fitted to data.
- **2 players only.** The knowledge and swap logic assume `NUM_PLAYERS = 2`; the
  real game supports up to six.
- **Pure Python.** ~0.12s per move at 1500 iterations.
