"""Read only requested native CIFAR-C arrays from the official tar via HTTP ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import requests

URLS={'cifar10':'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar',
      'cifar100':'https://zenodo.org/records/3555552/files/CIFAR-100-C.tar'}
WANTED={'gaussian_noise.npy','gaussian_blur.npy','contrast.npy','jpeg_compression.npy','labels.npy'}


def response(session,url,start,end):
    for attempt in range(6):
        result=session.get(url+f'?benchmark_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=60)
        if result.status_code==206 and result.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):return result
        result.close()
        time.sleep(min(2**attempt,30))
    raise RuntimeError(f'Official source did not return verified byte range {start}:{end}')


def run(dataset,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    session=requests.Session();offset=0;entries={};url=URLS[dataset]
    while True:
        with response(session,url,offset,offset+511) as r:header=r.content
        if header==bytes(512):break
        item=tarfile.TarInfo.frombuf(header,'utf-8','strict')
        name=Path(item.name).name
        if name in WANTED:
            entries[name]={'tar_name':item.name,'start':offset+512,'size':item.size,'official_source':url}
        offset+=512+((item.size+511)//512)*512
        if len(entries)==len(WANTED):break
    if set(entries)!=WANTED:raise RuntimeError('Official archive lacks requested files')
    (out/'archive_ranges.json').write_text(json.dumps(entries,indent=2))
    for name,entry in entries.items():
        destination=out/name;complete=out/(name+'_COMPLETE.json')
        if complete.exists() and destination.exists():continue
        temporary=out/(name+'.partial');failures=0
        # Small verified ranges tolerate Zenodo disconnects and resume existing partial files.
        while (temporary.stat().st_size if temporary.exists() else 0)<entry['size']:
            written=temporary.stat().st_size if temporary.exists() else 0
            end=min(entry['size'],written+16*1024*1024)-1
            try:
                with response(session,url,entry['start']+written,entry['start']+end) as r,temporary.open('ab') as stream:
                    for chunk in r.iter_content(1024*1024):stream.write(chunk)
            except requests.RequestException:
                failures+=1
                if failures>20:raise
                time.sleep(min(failures,15))
        if temporary.stat().st_size!=entry['size']:raise RuntimeError('Incomplete official range download')
        data=np.load(temporary,mmap_mode='r')
        if name!='labels.npy' and data.shape!=(50000,32,32,3):raise RuntimeError('Unexpected official corruption shape')
        shape=list(data.shape);del data
        checksum=hashlib.sha256()
        with temporary.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):checksum.update(chunk)
        temporary.rename(destination)
        complete.write_text(json.dumps({**entry,'sha256':checksum.hexdigest(),'shape':shape,'native_pixels':32}))
        print(json.dumps({'dataset':dataset,'downloaded':str(destination),'shape':shape}),flush=True)
    print(json.dumps({'dataset':dataset,'official_corruptions_ready':str(out)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=list(URLS));p.add_argument('--out',required=True)
    a=p.parse_args();run(a.dataset,a.out)
