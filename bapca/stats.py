"""
The statistics needed to say whether a difference is real.

Two things, both dependency-free:

  * Wilson score intervals, not "27 out of 40 = 67.5%". At n=40 the confidence
    interval is roughly +/-15 points, and a table of bare percentages invites
    everyone to over-read a gap that the data cannot support.

  * McNemar's exact test, because our conditions are PAIRED -- every condition
    sees the same items. Comparing two independent proportions here would throw
    away the pairing and lose most of the power we have. McNemar looks only at
    the items where the two conditions disagreed, which is exactly the evidence
    that one is better than the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Proportion:
    successes: int
    total: int

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson score interval: well behaved at small n and near 0 or 1."""
        n = self.total
        if n == 0:
            return (0.0, 0.0)
        p = self.rate
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    def __str__(self) -> str:
        low, high = self.wilson()
        return (f"{100 * self.rate:5.1f}%  [{100 * low:4.1f}, {100 * high:4.1f}]  "
                f"({self.successes}/{self.total})")


@dataclass
class McNemar:
    both: int          # both conditions correct
    only_a: int        # a correct, b wrong
    only_b: int        # b correct, a wrong
    neither: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b

    def verdict(self, alpha: float = 0.05) -> str:
        if self.discordant == 0:
            return "identical on every item"
        if self.p_value < alpha:
            better = "B" if self.only_b > self.only_a else "A"
            return f"different (p={self.p_value:.4f}), {better} better"
        return f"no significant difference (p={self.p_value:.4f})"


def mcnemar(a: list[bool], b: list[bool]) -> McNemar:
    """
    Exact two-sided McNemar test on paired outcomes.

    Under the null, each discordant pair is a fair coin, so the count of
    "b-only" wins is Binomial(n_discordant, 0.5). We compute the exact
    two-sided tail rather than the chi-square approximation, which is
    unreliable at the discordant counts a 40-item pilot produces.
    """
    if len(a) != len(b):
        raise ValueError(f"paired lists must match: {len(a)} vs {len(b)}")

    both = sum(1 for x, y in zip(a, b) if x and y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)

    n = only_a + only_b
    if n == 0:
        return McNemar(both, only_a, only_b, neither, 1.0)

    def binom_pmf(k: int) -> float:
        return math.comb(n, k) * 0.5 ** n

    observed = binom_pmf(only_b)
    # Two-sided exact: total probability of outcomes no more likely than ours.
    p = sum(binom_pmf(k) for k in range(n + 1) if binom_pmf(k) <= observed + 1e-12)
    return McNemar(both, only_a, only_b, neither, min(1.0, p))


def min_detectable_gap(n: int) -> float:
    """
    Roughly the smallest difference in rates worth believing at this n.

    Printed alongside the pilot so nobody reads a 5-point gap on 40 items as a
    finding. Half the width of a Wilson interval at p=0.5 is a fair rule of
    thumb for "do not interpret gaps smaller than this".
    """
    if n <= 0:
        return 1.0
    low, high = Proportion(n // 2, n).wilson()
    return (high - low) / 2


@dataclass
class Agreement:
    """How much two judges agree, beyond what chance would give."""
    n: int
    both_correct: int
    both_wrong: int
    a_only: int          # judge A said correct, judge B said wrong
    b_only: int
    kappa: float

    @property
    def raw(self) -> float:
        return (self.both_correct + self.both_wrong) / self.n if self.n else 0.0

    def reading(self) -> str:
        """Landis & Koch's conventional bands for kappa."""
        k = self.kappa
        if k < 0.0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"


def cohens_kappa(a: list[bool], b: list[bool]) -> Agreement:
    """
    Cohen's kappa for two binary raters.

    Raw agreement alone is misleading when one label dominates: two judges that
    both say "correct" 85% of the time agree ~75% by luck. Kappa subtracts that
    expectation, which is why LoCoMo-Plus reports judge agreement rather than
    raw match rates, and why we do too.
    """
    if len(a) != len(b):
        raise ValueError(f"paired lists must match: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return Agreement(0, 0, 0, 0, 0, 0.0)

    both_correct = sum(1 for x, y in zip(a, b) if x and y)
    both_wrong = sum(1 for x, y in zip(a, b) if not x and not y)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)

    observed = (both_correct + both_wrong) / n
    p_a, p_b = sum(a) / n, sum(b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    return Agreement(n, both_correct, both_wrong, a_only, b_only, kappa)
