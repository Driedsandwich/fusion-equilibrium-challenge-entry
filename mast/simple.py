#!/usr/bin/env python3
"""Label-free minimal MAST predictor; two explicit solenoid hypotheses."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'diag'), str(ROOT/'starter_ref')]
from capacity import read_columns, local_path, swap_reading
from greens import rectangle_flux, loop_flux
from data_fixes import align_mast_thomson_core
import argparse
import gc
import json
import time
import zipfile
import numpy as np
import pyarrow.parquet as pq

GEOMETRY = ['coil_input_column','coil_R','coil_Z','coil_width','coil_height','coil_angle1','coil_angle2']
P_NAMES = [f'magnetics_p{i}{side}_current' for i in range(2,7) for side in 'lu']
SOL = 'magnetics_sol_current'
INPUTS = ['efit_times','efit_grid_R','efit_grid_Z','magnetics_time',
          'magnetics_plasma_current','thomson_core_times','thomson_core_R',
          'thomson_core_Te','thomson_core_ne',SOL]+P_NAMES+GEOMETRY

def interp(t, values, target):
    ok = np.isfinite(t)&np.isfinite(values)
    if ok.sum()<2 or not np.all(np.diff(t[ok])>0):
        raise ValueError('Invalid current acquisition axis')
    if target.min()<t[ok][0] or target.max()>t[ok][-1]:
        raise ValueError('Refuse current extrapolation')
    return np.interp(target,t[ok],values[ok])*1000.

def coil_groups(row, order=3):
    rr, zz = np.meshgrid(row['efit_grid_R'],row['efit_grid_Z'])
    names = np.asarray(row['coil_input_column'])
    if set(names) != set(P_NAMES+[SOL]):
        raise ValueError('Unexpected poloidal geometry groups')
    columns = []; checks = {}
    for name in P_NAMES+[SOL]:
        idx = np.flatnonzero(names==name)
        area = row['coil_width'][idx]*row['coil_height'][idx]
        weights = area/area.sum()
        if (area<=0).any() or not np.isclose(weights.sum(),1):
            raise ValueError('Conductor allocation must sum to one')
        g = np.zeros(rr.shape)
        for j,w in zip(idx,weights):
            g += w*rectangle_flux(rr,zz,row['coil_R'][j],row['coil_Z'][j],
                                 row['coil_width'][j],row['coil_height'][j],
                                 row['coil_angle1'][j],row['coil_angle2'][j],order)
        columns.append(g.ravel())
        checks[name] = dict(conductors=len(idx), allocation_sum=float(weights.sum()))
    return np.column_stack(columns), checks

def thomson_geometry(row):
    R, Te, _ = align_mast_thomson_core(row)
    times = row['thomson_core_times']
    if Te.shape != (len(times),len(R)) or not np.all(np.diff(times)>0):
        raise ValueError('Unexpected Thomson layout')
    centers = []; radii = []; valid_times = []
    for t,profile in zip(times,Te):
        good = np.isfinite(R)&np.isfinite(profile)&(R>0)&(profile>0)
        if good.sum()<2:
            continue
        r, p = R[good], profile[good]
        active = r[p>.1*p.max()]
        if len(active)<2 or np.ptp(active)<=0:
            continue
        centers.append(float(np.sum(r*p)/p.sum()))
        radii.append(float(np.ptp(active)/2)); valid_times.append(t)
    if not valid_times:
        raise ValueError('No valid Thomson profile; no guessed MAST equilibrium prior')
    # Nearest measured profile; clamp at the acquisition ends, explicitly counted.
    target = row['efit_times']; ts = np.asarray(valid_times)
    right = np.minimum(np.searchsorted(ts,target),len(ts)-1)
    left = np.maximum(right-1,0)
    use = np.where(np.abs(ts[right]-target)<np.abs(ts[left]-target),right,left)
    return np.asarray(centers)[use],np.asarray(radii)[use],int(np.sum((target<ts[0])|(target>ts[-1])))

def predict(row, calibration, g):
    times = row['efit_times']; mt = row['magnetics_time']
    current = np.column_stack([interp(mt,row[n],times) for n in P_NAMES+[SOL]])
    ip = interp(mt,row['magnetics_plasma_current'],times)
    centers, radii, outside = thomson_geometry(row)
    rr,zz = np.meshgrid(row['efit_grid_R'],row['efit_grid_Z'])
    kappa = calibration['priors']['kappa']
    # Equal area polar quadrature: 3 radial rings x 16 angles, uniform ellipse.
    radius,theta = np.meshgrid(np.sqrt((np.arange(3)+.5)/3),
                              2*np.pi*(np.arange(16)+.137)/16)
    dx = (radius*np.cos(theta)).ravel(); dz = (radius*np.sin(theta)).ravel()
    plasma = np.empty((len(times),4225))
    for t,(center,a) in enumerate(zip(centers,radii)):
        ar = center+a*dx; bz = kappa*a*dz
        if (ar<=0).any():
            raise ValueError('Thomson ellipse reaches nonpositive source R')
        pg = loop_flux(rr.ravel()[:,None],zz.ravel()[:,None],ar,bz)
        plasma[t] = ip[t]*pg.mean(axis=1)
    vacuum_p = current[:,:10]@g[:,:10].T
    solenoid = current[:,-1,None]*g[:,-1]
    alpha = calibration['convention']['alpha_F']
    c = calibration['convention']['c_for_mast']
    outputs = {}
    for scale in (.5,1.):
        psi = (alpha*(vacuum_p+scale*solenoid+plasma)+c).reshape(len(times),65,65)
        outputs[scale] = dict(psirz=psi.astype(np.float16),
                             q95=np.full(len(times),calibration['priors']['q95'],np.float16),
                             betaN=np.full(len(times),calibration['priors']['betaN'],np.float16))
    return outputs,dict(Thomson_outside_time_frames=outside, R_center_median_m=float(np.median(centers)),
                        a_median_m=float(np.median(radii)),Ip_A_min=float(ip.min()),Ip_A_max=float(ip.max()))

def store_member(archive,name,array):
    with archive.open(name+'.npy','w',force_zip64=True) as member:
        np.lib.format.write_array(member,array,allow_pickle=False)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--no-guard',action='store_true')
    args = ap.parse_args(); start = time.monotonic()
    calibration = json.loads((ROOT/'diag/calib_joint_result.json').read_text())
    paths = sorted((ROOT/'_data/mast_public_test').glob('*.parquet'))
    if len(paths)!=10 or calibration['status']!='complete':
        raise ValueError('Require complete calibration and exactly ten local MAST shots')
    listing = json.loads((ROOT/'_data/_mast_listing_page1.json').read_text())
    ordered = [Path(x['path']).name for x in listing if x['type']=='file']
    stream_index = {name:i for i,name in enumerate(ordered)}
    out = ROOT/'mast'; out.mkdir(exist_ok=True)
    reports = {s:[] for s in (.5,1.)}; cache = None; amplitudes = {s:[] for s in reports}
    archives = {s:zipfile.ZipFile(out/f'mast_public_test_sol{s:g}.npz','w',compression=zipfile.ZIP_DEFLATED) for s in reports}
    try:
        for i,path in enumerate(paths):
            if not args.no_guard:
                mem = swap_reading()
                if not mem['ok'] or mem['free_mb']<300:
                    raise RuntimeError('Memory guard: '+str(mem))
            schema = pq.ParquetFile(path).schema_arrow.names
            if any(n.startswith('efit_') and n not in ('efit_times','efit_grid_R','efit_grid_Z') for n in schema):
                raise ValueError('MAST label column detected: refuse this file')
            row = read_columns(path,INPUTS)
            signature = [np.asarray(row[n]) for n in GEOMETRY+['efit_grid_R','efit_grid_Z']]
            if cache is None or not all(np.array_equal(a,b) for a,b in zip(cache,signature)):
                g,checks = coil_groups(row); cache = signature
            outputs,info = predict(row,calibration,g)
            index = stream_index[path.name]
            for scale,prediction in outputs.items():
                frame_count = len(row['efit_times'])
                for name,array in prediction.items():
                    expected = (frame_count,65,65) if name=='psirz' else (frame_count,)
                    if array.shape!=expected or array.dtype!=np.float16 or not np.isfinite(array).all():
                        raise ValueError(f'Invalid output {path.name} {name}')
                    store_member(archives[scale],f'shot_{index:04d}_{name}',array)
                psi = prediction['psirz'].astype(float)
                amp = np.ptp(psi,axis=(1,2)); amplitudes[scale].extend(amp)
                reports[scale].append(dict(file=path.name,stream_index=index,frames=frame_count,
                    shape=list(psi.shape),finite=True,dtype='float16',psi_min=float(psi.min()),psi_max=float(psi.max()),
                    positive_fraction=float(np.mean(psi>0)),negative_fraction=float(np.mean(psi<0)),
                    amplitude_median=float(np.median(amp)),**info))
            print(f'MAST {i+1}/10 stream={index} frames={frame_count}',flush=True)
            del row,outputs,prediction,psi
            gc.collect()
    finally:
        for archive in archives.values():
            archive.close()
    # Reopen the artifacts themselves, one member at a time.
    for scale in reports:
        with np.load(out/f'mast_public_test_sol{scale:g}.npz',allow_pickle=False) as archive:
            expected_keys = set()
            for shot in reports[scale]:
                for field in ('psirz','q95','betaN'):
                    key = f"shot_{shot['stream_index']:04d}_{field}"; expected_keys.add(key)
                    array = archive[key]
                    expected = (shot['frames'],65,65) if field=='psirz' else (shot['frames'],)
                    assert array.shape==expected and array.dtype==np.float16 and np.isfinite(array).all()
            assert set(archive.files)==expected_keys and len(expected_keys)==30
    summary = dict(status='complete',shots=10,frames=sum(s['frames'] for s in reports[.5]),
        order='original public-test stream indices from local listing; sparse ten-shot subset, not full submission',
        calibration=calibration['convention'],priors=calibration['priors'],group_checks=checks,
        solenoid='area-normalized geometry; unknown turns not inferred; coefficients 0.5 and 1 inside alpha_F',
        amplitude_definition='per-frame max(psi)-min(psi), then pooled median over every predicted frame',
        versions={str(s):dict(amplitude_median=float(np.median(amplitudes[s])),shots=reports[s]) for s in reports},
        artifact_reopen_validation=True,elapsed_seconds=time.monotonic()-start)
    (out/'simple_result.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('versions','group_checks')},ensure_ascii=False))

if __name__=='__main__':
    main()
