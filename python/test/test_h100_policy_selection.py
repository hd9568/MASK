from scripts.run_h100_sparsene_baseline import choose_calibrated


def candidate(name, median, valid=True):
    return {'name': name, 'median_ms': median, 'valid': valid}


def test_selection_requires_valid_baseline_and_ignores_invalid_candidates():
    assert choose_calibrated([candidate('baseline', None, False), candidate('multi-bind', 0.1)]) is None
    selected = choose_calibrated([candidate('baseline', 1.0), candidate('multi-bind', 0.1, False)])
    assert selected['name'] == 'baseline'


def test_selection_retains_baseline_inside_noise_band_and_accepts_real_gain():
    assert choose_calibrated([candidate('baseline', 1.0), candidate('multi-bind', 0.99)])['name'] == 'baseline'
    assert choose_calibrated([candidate('baseline', 1.0), candidate('multi-bind', 0.97)])['name'] == 'multi-bind'


def test_selection_uses_minimum_valid_median():
    items = [candidate('baseline', 1.0), candidate('auto-128', 0.91), candidate('auto-256', 0.89)]
    assert choose_calibrated(items)['name'] == 'auto-256'
