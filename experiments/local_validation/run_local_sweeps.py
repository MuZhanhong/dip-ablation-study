"""Local exploratory sweeps using unchanged repository model/training code.

Plan is dependency-free. Execution requires the repository requirements plus
matplotlib for the companion report. Existing attempts are never overwritten.
SSIM alone uses an explicitly recorded fixed 3x3 window so gap=4 is supported.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def source_hashes(repo):
    files = sorted((repo / 'src').glob('*.py')) + [repo / 'experiments/config.yaml', repo / 'baselines/fill.py']
    return {str(p.relative_to(repo)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def plan(args):
    args.batch.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in args.model_seeds:
        for network in [1, 2, 3, 5]:
            for gap in [4, 8, 16, 24, 32]:
                jobs.append(dict(sweep='gap_width', image_type='stripes', gap_size=gap,
                                 blur_scale=None, image_seed=0, model_seed=seed, network=network))
            for image_seed in args.image_seeds:
                for ell in [0, 1, 2, 4, 8]:
                    jobs.append(dict(sweep='blur_scale', image_type='blurred_noise', gap_size=16,
                                     blur_scale=ell, image_seed=image_seed, model_seed=seed, network=network))
    for i, job in enumerate(jobs):
        job.update(run_id=f'run_{i+1:04d}', channels=16, levels=4, steps=2000,
                   log_every=50, learning_rate=0.01, noise_scale=0.1,
                   size=64, heldout_fraction=0.05, heldout_seed=0, ssim_window=3,
                   selection='final' if job['network'] == 2 else 'heldout')
    manifest = dict(created_utc=stamp(), scope='local exploration, not team formal results',
                    status='draft; scope and dependency approval required before launch',
                    repository=str(args.repo.resolve()), source_sha256=source_hashes(args.repo),
                    jobs=jobs, planned_runs=len(jobs), notes=[
                        'Source: historical Week 2 plan; latest OneNote has not been provided.',
                        'Targets, input distribution, model, training mode and checkpoint rules follow current repository.',
                        'SSIM uses fixed win_size=3 for every run; not directly comparable to old win_size=7 SSIM.',
                        'No remote workflow is triggered by this script.'])
    save_json(args.batch / 'manifest.json', manifest)
    with (args.batch / 'run_matrix.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    print(f'Prepared {len(jobs)} jobs; no training started: {args.batch}', flush=True)


def get_manifest(args):
    manifest = json.loads((args.batch / 'manifest.json').read_text(encoding='utf-8'))
    repo = Path(manifest['repository'])
    if source_hashes(repo) != manifest['source_sha256']:
        raise RuntimeError('Repository source hashes changed; prepare a new manifest.')
    return manifest, repo


def execute_one(args, manifest, repo, job, benchmark=False):
    # Import only after the dependency-free plan has been created.
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity
    sys.path.insert(0, str(repo))
    from src.synthetic_data import make_image
    from src.masks import central_gap_mask, heldout_mask
    from src.models import DIPHourglass, ablation_config
    import src.training as training

    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but this PyTorch environment cannot use it.')
    torch.manual_seed(job['model_seed'])
    np.random.seed(job['model_seed'])
    root = args.batch / ('benchmarks' if benchmark else 'runs') / job['run_id']
    root.mkdir(parents=True, exist_ok=True)
    if not benchmark and any(root.glob('attempt_*/COMPLETE.json')):
        print('SKIP COMPLETE', job['run_id'], flush=True)
        return None
    attempt = next(root / f'attempt_{i:03d}' for i in range(1, 1000) if not (root / f'attempt_{i:03d}').exists())
    attempt.mkdir()
    old_ssim = training.ssim
    window = job['ssim_window']
    training.ssim = lambda p, t: float(structural_similarity(t, p, data_range=1.0, win_size=window))
    config = dict(job, benchmark=benchmark, threads=args.threads, device=args.device)
    save_json(attempt / 'config.json', config)
    environment = dict(start_utc=stamp(), python=sys.version, executable=sys.executable,
                       torch=torch.__version__, torch_cuda=torch.version.cuda,
                       numpy=np.__version__, platform=platform.platform(), cpu_count=os.cpu_count(),
                       device=args.device, gpu_name=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                       prediction_mode='training', source_sha256=manifest['source_sha256'])
    import scipy, skimage
    environment.update(scipy=scipy.__version__, skimage=skimage.__version__,
                       runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(attempt / 'environment.json', environment)
    print('START', job['run_id'], json.dumps(config), flush=True)
    start = time.perf_counter()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        image = make_image(job['image_type'], size=job['size'], seed=job['image_seed'],
                           blur_scale=1 if job['blur_scale'] is None else job['blur_scale'])
        gap = central_gap_mask(job['size'], job['gap_size'])
        heldout = heldout_mask(gap, job['heldout_fraction'], job['heldout_seed'])
        model = DIPHourglass(ablation_config(job['network'], channels=job['channels'], levels=job['levels']))
        param_count = sum(p.numel() for p in model.parameters())
        np.save(attempt / 'target.npy', image)
        np.savez_compressed(attempt / 'masks.npz', gap=gap, heldout=heldout)
        result = training.train_inpainting(image, gap, heldout, model, steps=job['steps'],
                    learning_rate=job['learning_rate'], log_every=job['log_every'],
                    noise_scale=job['noise_scale'], seed=job['model_seed'], device=args.device,
                    output_dir=attempt, selection=job['selection'])
        if args.device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        selected = next(row for row in result.history if int(row['step']) == result.selected_step)
        y, p = image[gap].astype(float), result.selected_prediction[gap].astype(float)
        correlation = float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-12 and y.std() > 1e-12 else None
        # gap SSIM window is fixed and recorded; no selection depends on SSIM.
        summary = dict(job, selected_step=result.selected_step, final_step=result.final_step,
                        selected_gap_mse=selected['gap_mse'], selected_heldout_mse=selected['heldout_mse'],
                        final_gap_mse=result.history[-1]['gap_mse'], gap_correlation=correlation,
                        prediction_mean=float(p.mean()), prediction_sd=float(p.std()),
                        target_mean=float(y.mean()), target_sd=float(y.std()),
                        parameter_count=param_count, runtime_seconds=seconds,
                        steps_per_second=job['steps']/seconds, finish_utc=stamp())
        if args.device == 'cuda':
            summary.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        assert np.isfinite(result.selected_prediction).all()
        if result.selection == 'heldout':
            assert result.best_heldout_mse == min(r['heldout_mse'] for r in result.history)
        else:
            assert np.array_equal(result.selected_prediction, result.final_prediction)
        assert not np.any(gap & heldout)
        assert result.final_step == job['steps']
        assert np.isclose(float(np.mean((p-y)**2)), selected['gap_mse'], rtol=1e-5, atol=1e-8)
        if source_hashes(repo) != manifest['source_sha256']:
            raise RuntimeError('Repository changed during run; outputs retained but not marked complete.')
        save_json(attempt / 'COMPLETE.json', summary)
        print('COMPLETE', job['run_id'], 'seconds', round(seconds, 2), 'gap_mse', selected['gap_mse'], flush=True)
        del model, result
        return summary
    except BaseException as error:
        save_json(attempt / 'FAILED.json', dict(time_utc=stamp(), error=repr(error)))
        raise
    finally:
        training.ssim = old_ssim
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def run(args):
    manifest, repo = get_manifest(args)
    if args.command == 'benchmark':
        # Cover all four architectures; results are timing checks, never sweep rows.
        summaries = []
        for network in [1, 2, 3, 5]:
            job = next(dict(j) for j in manifest['jobs'] if j['network'] == network and j['gap_size'] == 16 and j['image_type'] == 'stripes')
            job.update(run_id=f'benchmark_net{network}', steps=args.benchmark_steps, log_every=5)
            summaries.append(execute_one(args, manifest, repo, job, benchmark=True))
        estimates = {r['network']: r['runtime_seconds']/r['steps']*2000 for r in summaries}
        estimate = sum(estimates[j['network']] for j in manifest['jobs'])
        save_json(args.batch / 'timing_estimate.json', dict(benchmarks=summaries,
                  estimated_total_hours=estimate/3600, device=args.device,
                  note='Estimate includes benchmark startup and saving; longer runs may differ.'))
        print('Estimated serial hours:', round(estimate/3600, 2), flush=True)
    else:
        save_json(args.batch / 'launch.json', dict(time_utc=stamp(), scope='user-authorized local exploration',
                                                 threads=args.threads, device=args.device, planned_runs=len(manifest['jobs'])))
        count = 0
        for job in manifest['jobs']:
            result = execute_one(args, manifest, repo, job)
            if result is not None:
                count += 1
            if args.limit and count >= args.limit:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'benchmark', 'run'])
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--model-seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--image-seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--benchmark-steps', type=int, default=25)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'plan':
        if args.repo is None:
            parser.error('--repo is required for plan')
        if len(set(args.model_seeds)) != len(args.model_seeds) or len(set(args.image_seeds)) != len(args.image_seeds):
            parser.error('duplicate seeds')
        plan(args)
    else:
        if args.threads < 1:
            parser.error('--threads must be positive')
        run(args)


if __name__ == '__main__':
    main()
