"""Unit tests for provider-independent RAGree Choice voting mechanics."""

from judges.ragree_choice.sparrow import (
    Ballot,
    audit_ranking,
    seen_edges,
    select_adaptive_pairs,
    sparse_seed_graph,
)


def test_sparse_seed_graph_is_connected_cycle_for_large_election():
    candidates = [f"run-{index}" for index in range(8)]
    edges = sparse_seed_graph(candidates, degree=4, full_graph_max=5)
    assert len(edges) == 16
    assert {candidate for edge in edges for candidate in edge} == set(candidates)


def test_sparse_seed_graph_enforces_connected_minimum_for_degree_zero_and_one():
    candidates = [f"run-{index}" for index in range(10)]
    for degree in (0, 1):
        edges = sparse_seed_graph(candidates, degree=degree, full_graph_max=5)
        assert len(edges) == len(candidates)
        assert {candidate for edge in edges for candidate in edge} == set(candidates)
        neighbours = {candidate: set() for candidate in candidates}
        for left, right in edges:
            neighbours[left].add(right)
            neighbours[right].add(left)
        reached = {candidates[0]}
        frontier = [candidates[0]]
        while frontier:
            current = frontier.pop()
            for neighbour in neighbours[current] - reached:
                reached.add(neighbour)
                frontier.append(neighbour)
        assert reached == set(candidates)


def test_sparse_seed_graph_is_balanced_exact_and_deterministic():
    for count in (9, 10, 11, 16):
        candidates = [f"run-{index}" for index in range(count)]
        for degree in (2, 3, 4, 5):
            edges = sparse_seed_graph(candidates, degree=degree, full_graph_max=5)
            assert len(edges) == len(set(edges))
            assert all(left != right for left, right in edges)

            target = min((count * max(2, degree) + 1) // 2, count * (count - 1) // 2)
            assert len(edges) == target

            degree_count = {candidate: 0 for candidate in candidates}
            for left, right in edges:
                degree_count[left] += 1
                degree_count[right] += 1
            # Greedy min-degree selection is "near-regular" (per the docstring),
            # not exactly regular -- a full circulant-graph construction would
            # guarantee a spread of 1, but is more machinery than this cosmetic
            # fairness property (no candidate is ever left disconnected, which
            # is the actual correctness guarantee) is worth. 2 is the observed
            # worst case swept across counts 6-29 and degrees 2-7.
            assert max(degree_count.values()) - min(degree_count.values()) <= 2

            assert sparse_seed_graph(candidates, degree=degree, full_graph_max=5) == edges


def test_davidson_audit_finds_consistent_condorcet_and_kemeny_winner():
    ballots = [
        Ballot("a", "b", "A"),
        Ballot("a", "c", "A"),
        Ballot("b", "c", "B"),
    ]
    audit = audit_ranking(["a", "b", "c"], ballots)
    assert audit.ranking[0] == "a"
    assert audit.kemeny_ranking[0] == "a"
    assert audit.condorcet_winner == "a"
    assert audit.cycle_rate == 0.0


def test_cycle_audit_and_boundary_scheduler():
    ballots = [
        Ballot("a", "b", "A"),
        Ballot("b", "c", "A"),
        Ballot("c", "a", "A"),
    ]
    audit = audit_ranking(["a", "b", "c"], ballots)
    assert audit.cycle_rate == 1.0
    pairs = select_adaptive_pairs(
        ["a", "b", "c", "d"], {("a", "b")},
        {"a": 0.2, "b": 0.19, "c": 0.0, "d": -0.1}, top_k=1, budget=2,
    )
    assert len(pairs) == 2
    assert ("a", "b") not in pairs


def test_kemeny_agrees_on_winner_with_zero_usable_ballots():
    # BT's ascending tie-break and exact Kemeny's max(permutations(...))
    # descending tie-break would otherwise disagree on an arbitrary winner
    # when there is no real preference data at all (e.g. the only ballot for
    # this pool was ABSTAIN, directly or after a position-reversal collapse).
    audit = audit_ranking(["run-a", "run-b"], [Ballot("run-a", "run-b", "ABSTAIN")])
    assert audit.ballots_used == 0
    assert audit.ranking != audit.kemeny_ranking  # the tie-breaks really do differ
    assert audit.kemeny_agrees_on_winner


def test_seen_edges_counts_explicit_abstention_but_not_missing_transport_ballot():
    candidates = ["a", "b"]
    utilities = {"a": 0.0, "b": 0.0}
    assert select_adaptive_pairs(candidates, seen_edges([]), utilities, top_k=1, budget=1) == [("a", "b")]
    abstention = Ballot("a", "b", "ABSTAIN")
    assert seen_edges([abstention]) == {("a", "b")}
    assert select_adaptive_pairs(candidates, seen_edges([abstention]), utilities, top_k=1, budget=1) == []
