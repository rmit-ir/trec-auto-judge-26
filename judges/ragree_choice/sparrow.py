"""SPARROW's provider-independent preferential-ranking primitives.

The AutoJudge adapter supplies ballots from its injected LLM endpoint.  A
production Bedrock orchestrator can supply ballots from several model families
and assign calibrated weights without changing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import heapq
import math
from typing import Iterable, Mapping, Sequence


VALID_VOTES = frozenset({"A", "B", "TIE", "ABSTAIN"})


@dataclass(frozen=True)
class Ballot:
    """One hard preferential ballot for a pair of candidates."""

    candidate_a: str
    candidate_b: str
    vote: str
    judge_id: str = "injected-endpoint"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.candidate_a == self.candidate_b:
            raise ValueError("A ballot must compare two distinct candidates")
        if self.vote not in VALID_VOTES:
            raise ValueError(f"Unsupported ballot vote: {self.vote!r}")
        if self.weight <= 0:
            raise ValueError("Ballot weight must be positive")


@dataclass(frozen=True)
class RankingAudit:
    """A fitted online ranking plus voting-rule diagnostics."""

    utilities: Mapping[str, float]
    ranking: tuple[str, ...]
    kemeny_ranking: tuple[str, ...]
    condorcet_winner: str | None
    cycle_rate: float
    ballots_used: int

    @property
    def kemeny_agrees_on_winner(self) -> bool:
        """Whether the BT and Kemeny rankings pick the same winner.

        With zero usable ballots (e.g. the only ballot for this pool was an
        ABSTAIN, directly or after a position-reversal reconciliation), both
        rankings fall back to an all-tied ordering -- but BT's tie-break sorts
        candidate ids ascending while exact Kemeny's `max(permutations(...))`
        tie-break picks the lexicographically *greatest* permutation. Those two
        arbitrary, opposite tie-breaks would otherwise report a false
        disagreement with no data behind it, so treat "nothing to disagree
        about" as agreement rather than instability.
        """
        return not self.ranking or self.ballots_used == 0 or self.ranking[0] == self.kemeny_ranking[0]


def sparse_seed_graph(candidate_ids: Sequence[str], degree: int, full_graph_max: int) -> list[tuple[str, str]]:
    """Return a deterministic, near-regular connected graph of comparison edges.

    Small elections use all pairs. Larger elections always start with a complete
    cycle (every candidate gets a base degree of two -- or one, in the
    degenerate two-candidate cycle -- so no budget is ever too small to leave a
    candidate unevaluated), then top up to the requested degree
    by greedily adding the edge between whichever two not-yet-connected
    candidates currently have the lowest degree. Greedy min-degree selection
    (rather than exhausting one circular offset before starting the next) keeps
    the remaining edges spread evenly instead of favouring lexicographically
    early candidates when the budget runs out mid-pass. Degrees below two are
    accepted as a request for the minimum connected degree of two.
    """
    candidates = tuple(sorted(dict.fromkeys(candidate_ids)))
    count = len(candidates)
    if count < 2:
        return []
    if count <= full_graph_max:
        return list(combinations(candidates, 2))

    effective_degree = max(2, degree)
    target_edges = min(math.ceil(count * effective_degree / 2), count * (count - 1) // 2)
    # The cycle is non-negotiable: every candidate must have at least one edge
    # regardless of how small the requested budget is.
    edges = {
        tuple(sorted((candidates[index], candidates[(index + 1) % count])))
        for index in range(count)
    }
    if len(edges) >= target_edges:
        return sorted(edges)

    node_degree = {candidate: 2 for candidate in candidates}
    remaining = sorted(
        pair for pair in combinations(candidates, 2) if pair not in edges
    )
    while remaining and len(edges) < target_edges:
        best = min(remaining, key=lambda pair: (node_degree[pair[0]] + node_degree[pair[1]], pair))
        remaining.remove(best)
        edges.add(best)
        node_degree[best[0]] += 1
        node_degree[best[1]] += 1
    return sorted(edges)


def seen_edges(ballots: Iterable[Ballot]) -> set[tuple[str, str]]:
    """Return edges with a successfully parsed ballot, including ABSTAIN.

    Transport and parse failures are represented by the *absence* of a Ballot,
    so they cannot consume an adaptive scheduler slot.  An explicit ABSTAIN is
    a real, considered ballot and intentionally does consume that slot.
    """
    return {tuple(sorted((ballot.candidate_a, ballot.candidate_b))) for ballot in ballots}


def davidson_bt_fit(
    candidate_ids: Sequence[str],
    ballots: Iterable[Ballot],
    *,
    iterations: int = 160,
    learning_rate: float = 0.08,
    l2: float = 0.02,
    tie_factor: float = 1.0,
) -> dict[str, float]:
    """Fit a small, weighted Davidson--Bradley--Terry MAP approximation.

    Ties are modelled explicitly; abstentions deliberately provide no directional
    preference.  ``weight`` is where an external calibration service contributes
    judge reliability and independence discounts.
    """
    candidate_set = set(candidate_ids)
    usable = [
        ballot for ballot in ballots
        if ballot.vote != "ABSTAIN"
        and ballot.candidate_a in candidate_set
        and ballot.candidate_b in candidate_set
    ]
    utilities = {candidate: 0.0 for candidate in candidate_ids}
    if not usable:
        return utilities

    for _ in range(iterations):
        gradients = {candidate: -l2 * value for candidate, value in utilities.items()}
        for ballot in usable:
            delta = max(-30.0, min(30.0, utilities[ballot.candidate_a] - utilities[ballot.candidate_b]))
            win_a = math.exp(delta)
            tie = tie_factor * math.exp(delta / 2.0)
            denominator_gradient = (win_a + 0.5 * tie) / (win_a + 1.0 + tie)
            if ballot.vote == "A":
                derivative = 1.0 - denominator_gradient
            elif ballot.vote == "B":
                derivative = -denominator_gradient
            else:  # TIE
                derivative = 0.5 - denominator_gradient
            gradients[ballot.candidate_a] += ballot.weight * derivative
            gradients[ballot.candidate_b] -= ballot.weight * derivative

        for candidate in utilities:
            utilities[candidate] += learning_rate * gradients[candidate]
        centre = sum(utilities.values()) / len(utilities)
        for candidate in utilities:
            utilities[candidate] -= centre
    return utilities


def davidson_win_probability(utility_a: float, utility_b: float, tie_factor: float = 1.0) -> float:
    """Return the Davidson-model probability that A beats B (excluding no data)."""
    delta = max(-30.0, min(30.0, utility_a - utility_b))
    win_a = math.exp(delta)
    tie = tie_factor * math.exp(delta / 2.0)
    return win_a / (win_a + 1.0 + tie)


def weighted_margins(candidate_ids: Sequence[str], ballots: Iterable[Ballot]) -> dict[tuple[str, str], float]:
    """Build the directed weighted preference matrix used by voting audits."""
    margins = {(left, right): 0.0 for left in candidate_ids for right in candidate_ids if left != right}
    candidate_set = set(candidate_ids)
    for ballot in ballots:
        if ballot.vote in {"TIE", "ABSTAIN"}:
            continue
        if ballot.candidate_a not in candidate_set or ballot.candidate_b not in candidate_set:
            continue
        winner, loser = (
            (ballot.candidate_a, ballot.candidate_b)
            if ballot.vote == "A"
            else (ballot.candidate_b, ballot.candidate_a)
        )
        margins[(winner, loser)] += ballot.weight
        margins[(loser, winner)] -= ballot.weight
    return margins


def condorcet_winner(candidate_ids: Sequence[str], margins: Mapping[tuple[str, str], float]) -> str | None:
    """Return a strict Condorcet winner when the observed matrix has one."""
    for candidate in candidate_ids:
        if all(margins[(candidate, other)] > 0 for other in candidate_ids if other != candidate):
            return candidate
    return None


def cycle_rate(candidate_ids: Sequence[str], margins: Mapping[tuple[str, str], float]) -> float:
    """Fraction of candidate triples that form a directed preference cycle."""
    triples = list(combinations(candidate_ids, 3))
    if not triples:
        return 0.0
    cycles = 0
    for first, second, third in triples:
        forward = margins[(first, second)] > 0 and margins[(second, third)] > 0 and margins[(third, first)] > 0
        backward = margins[(second, first)] > 0 and margins[(third, second)] > 0 and margins[(first, third)] > 0
        cycles += int(forward or backward)
    return cycles / len(triples)


def _kemeny_score(order: Sequence[str], margins: Mapping[tuple[str, str], float]) -> float:
    return sum(margins[(higher, lower)] for index, higher in enumerate(order) for lower in order[index + 1:])


def kemeny_ranking(candidate_ids: Sequence[str], margins: Mapping[tuple[str, str], float], utilities: Mapping[str, float], exact_limit: int = 8) -> tuple[str, ...]:
    """Return exact Kemeny order for small elections, local-search heuristic otherwise."""
    candidates = tuple(candidate_ids)
    if len(candidates) <= 1:
        return candidates
    if len(candidates) <= exact_limit:
        return max(permutations(candidates), key=lambda order: (_kemeny_score(order, margins), order))

    order = list(sorted(candidates, key=lambda candidate: (-utilities[candidate], candidate)))
    improved = True
    while improved:
        improved = False
        current_score = _kemeny_score(order, margins)
        for index in range(len(order) - 1):
            proposal = order[:]
            proposal[index], proposal[index + 1] = proposal[index + 1], proposal[index]
            proposal_score = _kemeny_score(proposal, margins)
            if proposal_score > current_score:
                order, current_score, improved = proposal, proposal_score, True
    return tuple(order)


def audit_ranking(candidate_ids: Sequence[str], ballots: Iterable[Ballot], exact_kemeny_limit: int = 8) -> RankingAudit:
    """Fit the operational model and its independent Condorcet/Kemeny audits."""
    materialized = list(ballots)
    utilities = davidson_bt_fit(candidate_ids, materialized)
    ranking = tuple(sorted(candidate_ids, key=lambda candidate: (-utilities[candidate], candidate)))
    margins = weighted_margins(candidate_ids, materialized)
    return RankingAudit(
        utilities=utilities,
        ranking=ranking,
        kemeny_ranking=kemeny_ranking(candidate_ids, margins, utilities, exact_kemeny_limit),
        condorcet_winner=condorcet_winner(candidate_ids, margins),
        cycle_rate=cycle_rate(candidate_ids, margins),
        ballots_used=sum(ballot.vote != "ABSTAIN" for ballot in materialized),
    )


def select_reversal_ballots(
    candidate_ids: Sequence[str],
    ballots: Sequence[Ballot],
    utilities: Mapping[str, float],
    *,
    top_k: int,
    budget: int,
) -> list[Ballot]:
    """Pick up to `budget` already-cast ballots worth re-checking with the
    answer order swapped: directional (A/B) ballots whose pair crosses the
    top-k utility boundary first, then TIE ballots. ABSTAIN ballots are never
    reversal-checked -- there is no direction to validate. This deliberately
    does not reverse every pair (see the design doc's caution that doing so
    fixes position bias while doing nothing for verbosity bias, at double the
    cost); it targets the pairs where a swapped answer is most likely to
    change the decision that matters.
    """
    if budget <= 0:
        return []
    ranked = sorted(candidate_ids, key=lambda candidate: (-utilities.get(candidate, 0.0), candidate))
    membership = set(ranked[:max(1, top_k)])
    boundary: list[Ballot] = []
    ties: list[Ballot] = []
    seen: set[tuple[str, str]] = set()
    for ballot in ballots:
        edge = tuple(sorted((ballot.candidate_a, ballot.candidate_b)))
        if edge in seen or ballot.vote == "ABSTAIN":
            continue
        seen.add(edge)
        if ballot.vote == "TIE":
            ties.append(ballot)
        elif (ballot.candidate_a in membership) != (ballot.candidate_b in membership):
            boundary.append(ballot)
    return (boundary + ties)[:budget]


def select_adaptive_pairs(
    candidate_ids: Sequence[str],
    seen_edges: set[tuple[str, str]],
    utilities: Mapping[str, float],
    *,
    top_k: int,
    budget: int,
    expected_cost: float = 1.0,
) -> list[tuple[str, str]]:
    """Use a max-heap to prioritise uncertain, top-k-boundary comparisons."""
    if budget <= 0:
        return []
    ranked = sorted(candidate_ids, key=lambda candidate: (-utilities[candidate], candidate))
    membership = set(ranked[:max(1, top_k)])
    heap: list[tuple[float, str, str]] = []
    for candidate_a, candidate_b in combinations(sorted(candidate_ids), 2):
        edge = (candidate_a, candidate_b)
        if edge in seen_edges:
            continue
        boundary_bonus = 2.0 if (candidate_a in membership) != (candidate_b in membership) else 0.0
        uncertainty = 1.0 / (1.0 + abs(utilities[candidate_a] - utilities[candidate_b]))
        priority = (boundary_bonus + uncertainty) / max(expected_cost, 1e-9)
        heapq.heappush(heap, (-priority, candidate_a, candidate_b))
    selected: list[tuple[str, str]] = []
    while heap and len(selected) < budget:
        _priority, candidate_a, candidate_b = heapq.heappop(heap)
        selected.append((candidate_a, candidate_b))
    return selected
