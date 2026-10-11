"""Component timings of the frozen full CIFAR Grounder pipeline, unchanged math."""
import argparse,json,time,os,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from benchmark.pini import import_core
from benchmark.pini_cifar_audit import PHRASES


def audit_kernels(batch=4,calls=8):
    import_core();torch.set_num_threads(4)
    from spatial_noise.grounding import FrozenGrounder
    from transformers.integrations.hub_kernels import kernelize,register_kernel_mapping_transformers
    detector=FrozenGrounder(phrases=PHRASES,model_path='/home/pengcheng/Pi-CBM/D-Pi/artifacts/models/grounding-dino-tiny',image_size=800,threshold=.25,precision='fp32')
    def weights():
        h=hashlib.sha256()
        for key,value in detector.detector.state_dict().items():h.update(key.encode());h.update(value.cpu().numpy().tobytes())
        return h.hexdigest()
    original_sha=weights();rgb=np.load('artifacts/pini_matched/cifar10/cache/rgb_original.npy',mmap_mode='r');reference=[];timing={};peaks={};errors={k:[] for k in ['features','maps','boxes','scores']};valid_equal=[];map_support_equal=[];forward_modules=[]
    for mode in ['original_torch_fallback','official_Hub_CUDA']:
        if mode=='official_Hub_CUDA':
            register_kernel_mapping_transformers()
            kernelize(detector.detector)
            assert weights()==original_sha
            forward_modules=[m.forward.__func__.__module__ for m in detector.detector.modules() if getattr(type(m),'kernel_layer_name',None)=='MultiScaleDeformableAttention']
            assert forward_modules and all('transformers.models' not in m for m in forward_modules),forward_modules
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();rows=[]
        for i in range(calls):
            images=[Image.fromarray(np.clip(x.transpose(1,2,0),0,255).astype(np.uint8)) for x in rgb[i*batch:(i+1)*batch]]
            start=time.perf_counter();r=detector(images);arrays={k:r[k].cpu().numpy().copy() for k in ['features','maps','valid','boxes','scores']};total=time.perf_counter()-start
            rows.append({'total_s':total,'GPU_forward_s':r['detector_forward_s']})
            if mode=='original_torch_fallback':reference.append(arrays)
            else:
                for key in errors:errors[key].append(float(np.max(np.abs(arrays[key].astype(float)-reference[i][key].astype(float)))))
                valid_equal.append(bool(np.array_equal(arrays['valid'],reference[i]['valid'])))
                map_support_equal.append(bool(np.array_equal(arrays['maps']>0,reference[i]['maps']>0)))
                np.savez_compressed(Path('artifacts/cifar_feasibility')/('kernel_sample_'+str(i)+'.npz'),ids=np.arange(i*batch,(i+1)*batch),**{'original_'+k:v for k,v in reference[i].items()},**{'optimized_'+k:v for k,v in arrays.items()})
        timing[mode]=rows;peaks[mode]=torch.cuda.max_memory_allocated()
    out={'weights_sha256':original_sha,'weight_SHA_identical':True,'images':batch*calls,'batch':batch,'image_size':800,'threshold':.25,'precision':'fp32','timing':timing,'peak_allocated_bytes':peaks,'maximum_abs_difference':{k:max(v) for k,v in errors.items()},'valid_all_identical':all(valid_equal),'kernel_source':'kernels-community/deformable-detr version1 via official transformers kernelize','original_view_not_modified':True}
    out.update(kernel_actually_enabled=True,attention_forward_modules=sorted(set(forward_modules)),positive_map_support_all_identical=all(map_support_equal),mean_total_seconds_excluding_first={k:float(np.mean([r['total_s'] for r in v[1:]])) for k,v in timing.items()})
    Path('artifacts/cifar_feasibility/kernel_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);detector.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=4);p.add_argument('--calls',type=int,default=8);p.add_argument('--kernels',action='store_true');a=p.parse_args()
    if a.kernels:return audit_kernels(a.batch,a.calls)
    torch.set_num_threads(4);import_core()
    from spatial_noise.grounding import FrozenGrounder
    detector=FrozenGrounder(phrases=PHRASES,model_path='/home/pengcheng/Pi-CBM/D-Pi/artifacts/models/grounding-dino-tiny',image_size=800,threshold=.25,precision='fp32')
    processor=detector.processor;timings=[]
    class TimedProcessor:
        def __call__(self,*args,**kwargs):
            start=time.perf_counter();r=processor(*args,**kwargs);timings.append(time.perf_counter()-start);return r
    detector.processor=TimedProcessor()
    rgb=np.load('artifacts/pini_matched/cifar10/cache/rgb_original.npy',mmap_mode='r');rows=[]
    for i in range(a.calls):
        start=time.perf_counter();images=[Image.fromarray(np.clip(x.transpose(1,2,0),0,255).astype(np.uint8)) for x in rgb[i*a.batch:(i+1)*a.batch]];pil=time.perf_counter()-start
        start=time.perf_counter();r=detector(images);grounder=time.perf_counter()-start
        start=time.perf_counter();arrays={k:r[k].cpu().numpy() for k in ['features','maps','valid','boxes','scores']};transfer=time.perf_counter()-start
        start=time.perf_counter()
        # Same 14->224 nearest-grid union used solely for diagnostic mask.
        foreground=(arrays['maps'].max(1)>0).repeat(16,1).repeat(16,2)
        post=time.perf_counter()-start
        rows.append({'PIL_quantization_and_rgb_read_s':pil,'processor_CPU_s':timings[-1],'GPU_forward_s':r['detector_forward_s'],'grounder_total_s':grounder,'other_grounder_transfer_and_postprocess_s':grounder-timings[-1]-r['detector_forward_s'],'outputs_GPU_to_CPU_s':transfer,'diagnostic_union_mask_CPU_s':post,'total_without_disk_flush_s':pil+grounder+transfer+post})
    out={'pid':os.getpid(),'images':a.batch*a.calls,'batch':a.batch,'image_size':800,'threshold':.25,'precision':'fp32','math_unchanged':True,'timings':rows,'mean_seconds':{k:float(np.mean([x[k] for x in rows[1:]])) for k in rows[0]},'warmup_first_call_excluded_from_means':True,'concurrent_GPU_jobs':'record nvidia-smi separately; queue contention affects measured GPU latency','disk_IO':'source RGB memory map; this pilot does not flush final cache files, inspect actual pipeline throughput separately'}
    path=Path('artifacts/cifar_feasibility/pipeline_profile.json');path.write_text(json.dumps(out,indent=2));print(json.dumps(out['mean_seconds']),flush=True);detector.close()

if __name__=='__main__':main()
