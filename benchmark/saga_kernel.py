"""Opt-in runtime contraction substitution; frozen upstream files stay intact."""
import contextlib
import hashlib
import inspect

import torch


@contextlib.contextmanager
def gradient_kernel(module, mode='upstream'):
    if mode not in ('upstream', 'matmul'):
        raise ValueError(f'Unknown SAGA gradient kernel: {mode}')
    original = module.train_saga
    source = inspect.getsource(original)
    replacement = source
    substitutions = {
        '(a.unsqueeze(2) * X.unsqueeze(1)).mean(0)': '(a.T @ X) / X.size(0)',
        '(a_prev.unsqueeze(2) * X.unsqueeze(1)).mean(0)': '(a_prev.T @ X) / X.size(0)',
    }
    for before, after in substitutions.items():
        if source.count(before) != 1:
            raise RuntimeError('Upstream SAGA source changed; refuse an unreviewed kernel substitution')
        replacement = replacement.replace(before, after)
    metadata = {
        'name': mode, 'optional_benchmark_runtime_wrapper': mode != 'upstream',
        'original_function_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'replacement_function_source_sha256': hashlib.sha256(replacement.encode()).hexdigest(),
        'replacement_sites': 2 if mode == 'matmul' else 0,
        'core_file_modified': False,
        'numeric_audit': 'artifacts/audit/saga_outer_gemm_cpu.json',
        'cuda_numeric_and_training_audit': 'artifacts/audit/optional_head_regression.json',
    }
    if mode == 'upstream':
        yield metadata
        return
    namespace = dict(original.__globals__)
    exec(compile(replacement, '<benchmark-only SAGA matmul contraction>', 'exec'), namespace)
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        module.train_saga = namespace['train_saga']
        metadata.update(float32_matmul_precision='highest', matmul_allow_tf32=False,
                        cudnn_allow_tf32=False)
        yield metadata
    finally:
        module.train_saga = original
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
