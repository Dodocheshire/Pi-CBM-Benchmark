"""CPU numerical audit only; does not install or monkeypatch a SAGA solver."""
import hashlib
import json
import sys
import zipfile

import numpy as np
import torch

from .task_backbones import ROOT, file_sha


def difference(a, b):
    return {'max_abs': float((a-b).abs().max()),
            'relative_l2': float((a-b).norm()/a.norm().clamp_min(1e-30))}


def main():
    torch.set_num_threads(3)
    torch.manual_seed(42)
    source = ROOT / 'artifacts/matched/lf/imagenet_200_per_class/rn18_imagenet/seed42/single_draw/lfpi_global_train_original.npz'
    with zipfile.ZipFile(source) as archive:
        with archive.open('concepts.npy') as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream)
                                    if version == (1, 0) else np.lib.format.read_array_header_2_0(stream))
            assert not fortran and dtype == np.dtype('float32')
            offset = stream.tell()
            ids = np.linspace(0, shape[0]-1, 16, dtype=np.int64)
            rows = []
            for index in ids:
                stream.seek(offset + int(index)*shape[1]*dtype.itemsize)
                rows.append(np.frombuffer(stream.read(shape[1]*dtype.itemsize), dtype=dtype).copy())
    with np.load(source) as values:
        labels = values['labels'][ids].copy()
        subject_ids = values['ids'][ids].copy()
    raw = torch.from_numpy(np.stack(rows))
    core = ROOT / 'methods/LF-Pi'
    sys.path.insert(0, str(core))
    from pi_cbm.vendor.lfcbm.elasticnet import soft_threshold_with_shrinkage
    batch, concepts = raw.shape
    classes = 1000
    lr, lam, alpha = .1, .0007, .99
    records = []
    for dtype in (torch.float64, torch.float32):
        torch.manual_seed(42)
        x = raw.to(dtype)
        weight = torch.randn(classes, concepts, dtype=dtype)*.001
        bias = torch.randn(classes, dtype=dtype)*.001
        target = torch.nn.functional.one_hot(torch.from_numpy(labels).long(), classes).to(dtype)
        a = torch.softmax(torch.nn.functional.linear(x, weight, bias), -1) - target
        previous = torch.softmax(torch.randn(batch, classes, dtype=dtype), -1) - target
        average = torch.randn(classes, concepts, dtype=dtype)*.001
        direct = (a.unsqueeze(2)*x.unsqueeze(1)).mean(0)
        direct_previous = (previous.unsqueeze(2)*x.unsqueeze(1)).mean(0)
        gemm = a.T @ x / batch
        gemm_previous = previous.T @ x / batch
        saga = direct-direct_previous+average
        saga_gemm = gemm-gemm_previous+average
        threshold = lr*lam*alpha
        shrink = lr*lam*(1-alpha)
        old = soft_threshold_with_shrinkage(weight-lr*saga, threshold, shrink)
        new = soft_threshold_with_shrinkage(weight-lr*saga_gemm, threshold, shrink)
        # Deliberately put coordinates on the old pre-prox threshold. This is a
        # boundary sensitivity probe, not a second trained model or optimizer run.
        edge_weight = lr*saga + threshold
        edge_old = soft_threshold_with_shrinkage(edge_weight-lr*saga, threshold, shrink)
        edge_new = soft_threshold_with_shrinkage(edge_weight-lr*saga_gemm, threshold, shrink)
        connection_weight = lr*saga + threshold + 1e-5*(1+shrink)
        connection_old = soft_threshold_with_shrinkage(connection_weight-lr*saga, threshold, shrink)
        connection_new = soft_threshold_with_shrinkage(connection_weight-lr*saga_gemm, threshold, shrink)
        avg_old = average+(direct-direct_previous)*batch/shape[0]
        avg_new = average+(gemm-gemm_previous)*batch/shape[0]
        records.append({'dtype': str(dtype), 'new_scalar_gradient': difference(direct, gemm),
            'previous_scalar_gradient': difference(direct_previous, gemm_previous),
            'saga_direction': difference(saga, saga_gemm), 'one_step_upstream_prox': difference(old, new),
            'gradient_average_update': difference(avg_old, avg_new),
            'regular_weight_exact_zero_mask_changes': int(((old==0)!=(new==0)).sum()),
            'regular_weight_1e_5_connection_mask_changes': int(((old.abs()>1e-5)!=(new.abs()>1e-5)).sum()),
            'boundary_probe_exact_zero_mask_changes': int(((edge_old==0)!=(edge_new==0)).sum()),
            'boundary_probe_1e_5_connection_mask_changes': int(((edge_old.abs()>1e-5)!=(edge_new.abs()>1e-5)).sum()),
            'boundary_probe_max_update_difference': float((edge_old-edge_new).abs().max()),
            'connection_1e_5_boundary_probe_mask_changes': int(((connection_old.abs()>1e-5)!=(connection_new.abs()>1e-5)).sum()),
            'connection_boundary_probe_max_update_difference': float((connection_old-connection_new).abs().max()),
            'bias_update_difference': 0.0, 'table_update_difference': 0.0,
            'prox_threshold': threshold, 'prox_shrink': shrink})
    audit = {'deployed': False, 'device': 'cpu', 'gpu_used': False,
        'source': str(source), 'source_sha256': file_sha(source), 'source_array_shape': list(shape),
        'source_field': 'concepts', 'sample_row_positions': ids.tolist(), 'sample_subject_ids': subject_ids.tolist(),
        'sample_labels': labels.tolist(), 'sample_byte_sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        'feature_preprocessing': 'raw actual export values, no audit-only standardization; both formulas receive exactly the same X',
        'classes': classes, 'batch': batch, 'concepts': concepts, 'seed': 42,
        'upstream': str(core/'pi_cbm/vendor/lfcbm/elasticnet.py'),
        'upstream_sha256': file_sha(core/'pi_cbm/vendor/lfcbm/elasticnet.py'),
        'original_formula': '(a.unsqueeze(2)*X.unsqueeze(1)).mean(0)',
        'candidate_formula': 'a.T @ X / X.shape[0]', 'records': records,
        'original_batch256_outer_temporary_GiB': 256*classes*concepts*4/2**30,
        'candidate_float32_CK_output_GiB': classes*concepts*4/2**30,
        'cpu_audit_outer_temporary_float64_GiB': batch*classes*concepts*8/2**30,
        'mathematical_equivalence': 'identical contraction sum over batch in exact arithmetic, for both a and a_prev including already-applied sample weights',
        'numerical_limitations': 'reduction order differs; threshold ties can change exact zero masks; CPU audit does not establish CUDA/TF32 parity or unchanged full-path convergence/validation choice',
        'cuda_followup_requirement': 'audit CUDA reduction and GEMM with explicitly recorded matmul precision; TF32 may make differences larger than CPU float32; retain full precision for this contraction if the unchanged solver protocol requires it'}
    output = ROOT / 'artifacts/audit/saga_outer_gemm_cpu.json'
    output.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
