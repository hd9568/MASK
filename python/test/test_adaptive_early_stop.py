from sparsene.op_gen.early_stop import stop_decision


def row(i, value, spread=0.1, status='validated'):
    return {'iteration': i, 'time_ms': value, 'spread_pct': spread, 'status': status}


def test_waits_for_minimum_then_stops_on_stagnation():
    assert not stop_decision([row(0, 10), row(1, 10)])['stop']
    assert stop_decision([row(0, 10), row(1, 9.95), row(2, 9.94)])['stop']


def test_significant_improvement_resets_patience():
    assert not stop_decision([row(0, 10), row(1, 9.95), row(2, 9.5)])['stop']


def test_failures_never_count_as_stagnation():
    assert not stop_decision([row(0, 10), row(1, None, status='failed'), row(2, 10)])['stop']


def test_measurement_spread_sets_conservative_noise_floor():
    out = stop_decision([row(0, 10, 4), row(1, 9.7), row(2, 9.65)])
    assert out['stop'] and out['steps'][1]['required_gain'] == 0.04
