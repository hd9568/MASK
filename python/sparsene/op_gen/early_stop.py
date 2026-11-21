"""Conservative, auditable early stopping for correctness-gated searches."""
from __future__ import annotations

import math
from typing import Any


def stop_decision(observations: list[dict[str, Any]], *, min_valid: int = 3,
                  patience: int = 2, minimum_gain: float = 0.015) -> dict[str, Any]:
    """Each item has status, time_ms, spread_pct and iteration.

    Failures/invalid runs never count as stagnation or supply an incumbent.
    The absolute spread of either timing batch is a conservative noise floor.
    """
    best = None
    best_spread = 0.0
    valid = stale = 0
    steps = []
    for row in observations:
        val = row.get('time_ms')
        good = (row.get('status') == 'validated' and isinstance(val, (int, float))
                and math.isfinite(val) and val > 0)
        if not good:
            stale = 0
            steps.append({'iteration': row.get('iteration'), 'outcome': 'excluded_invalid'})
            continue
        valid += 1
        spread = row.get('spread_pct')
        spread = max(0.0, float(spread)) / 100 if isinstance(spread, (int, float)) and math.isfinite(spread) else 0.0
        threshold = max(minimum_gain, spread, best_spread)
        gain = (best - val) / best if best is not None else None
        improved = best is None or gain > threshold
        if improved:
            best, best_spread, stale = float(val), spread, 0
        else:
            stale += 1
        steps.append({'iteration': row.get('iteration'), 'outcome': 'improved' if improved else 'no_significant_improvement',
                      'relative_gain': gain, 'required_gain': threshold})
    stop = valid >= min_valid and stale >= patience
    return {'stop': stop, 'reason': 'stagnation' if stop else 'continue', 'valid_observations': valid,
            'consecutive_no_improvement': stale, 'min_valid': min_valid, 'patience': patience,
            'minimum_gain': minimum_gain, 'best_significant_ms': best, 'steps': steps}
