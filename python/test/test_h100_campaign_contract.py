from argparse import Namespace
from pathlib import Path
import pytest
from run_h100_ds_v4pro_all26 import DATASETS, worker_command
from run_v7_strategy_feedback import _call_strategy_with_retry

def test_h100_campaign_budget_and_no_historical_feedback():
    a=Namespace(repo=Path('/repo'),run_dir=Path('/run'),feature_dir=Path('/features'),incumbent_dir=Path('/empty'),model_config=Path('/config'),ncu_bin=Path('/ncu'))
    c=worker_command(a,'ddi')
    assert len(DATASETS)==26
    assert c[c.index('--format-iterations')+1]=='3'
    assert c[c.index('--strategy-agent-iterations')+1]=='5'
    assert c[c.index('--format-max-attempts')+1]=='1'
    assert c[c.index('--initial-format')+1]=='sparsene_historical_best'
    assert '--historical-format-only' in c
    assert '--initial-feedback-run-dir' not in c
    assert '--skip-incumbent-update' in c

def test_h100_mode_never_replays_ambiguous_strategy_errors(monkeypatch,tmp_path):
    monkeypatch.setenv('SPARSEKG_NO_AMBIGUOUS_RETRY','1')
    calls=[]
    def request():
        calls.append(1)
        raise TimeoutError('ambiguous outcome')
    with pytest.raises(TimeoutError):
        _call_strategy_with_retry(request,audit_path=tmp_path/'audit.json')
    assert len(calls)==1

def test_safe_mode_retries_refused_connection_only(monkeypatch):
    from urllib.error import URLError
    from sparsene.agent import client
    monkeypatch.setenv('SPARSEKG_NO_AMBIGUOUS_RETRY','1')
    calls=[]
    def open_url(*args,**kwargs):
        calls.append(1)
        if len(calls)<3: raise URLError(ConnectionRefusedError('refused'))
        return 'ok'
    monkeypatch.setattr(client.urllib_request,'urlopen',open_url)
    monkeypatch.setattr(client.time,'sleep',lambda _: None)
    assert client._urlopen_with_tls_eof_retry(object(),1)=='ok'
    assert len(calls)==3


def test_safe_mode_does_not_retry_unknown_tls_failure(monkeypatch):
    import ssl
    from urllib.error import URLError
    from sparsene.agent import client
    monkeypatch.setenv('SPARSEKG_NO_AMBIGUOUS_RETRY','1')
    calls=[]
    def open_url(*args,**kwargs):
        calls.append(1)
        raise URLError(ssl.SSLError('UNEXPECTED_EOF_WHILE_READING'))
    monkeypatch.setattr(client.urllib_request,'urlopen',open_url)
    with pytest.raises(URLError): client._urlopen_with_tls_eof_retry(object(),1)
    assert len(calls)==1

def test_hardware_probe_falls_back_to_unknown_resources(monkeypatch):
    import ctypes
    from sparsene.op_gen import strategy_agent
    monkeypatch.setenv('SPARSEKG_CUDA_ARCH', '90')
    monkeypatch.delenv('CUDA_ARCH', raising=False)
    monkeypatch.setattr(strategy_agent, '_safe_read_command_output', lambda _: 'NVIDIA H100 PCIe, 610.43, 81559, 9.0')
    def unavailable(*args):
        raise OSError('no CUDA library')
    monkeypatch.setattr(ctypes, 'CDLL', unavailable)
    profile = strategy_agent.discover_hardware_profile()
    assert profile.cuda_arch == 'sm_90'
    assert profile.gpu_name == 'NVIDIA H100 PCIe'
    assert profile.shared_memory_per_sm_kb is None
    assert profile.sm_count is None
