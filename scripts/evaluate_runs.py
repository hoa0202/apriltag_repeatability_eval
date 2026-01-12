#!/usr/bin/env python3
"""
평가 스크립트: 10회 run 궤적 비교 (ATE/CTE 분석)

Usage:
    # CLI 모드
    python3 evaluate_runs.py --ref ./data/run_01.csv --runs ./data/run_*.csv --ds 0.01 --out_dir ./out
    
    # 직접 실행 모드 (아래 CONFIG 섹션 수정 후)
    python3 evaluate_runs.py
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import List, Dict, Tuple, Optional
import json
import os

# ============================================================================
# CONFIG: 여기서 파라미터 수정 (CLI 인자 없이 실행할 때 사용됨)
# ============================================================================
CONFIG = {
    'ref': './data/run_01.csv',           # 기준 run CSV 경로
    'runs': './data/run_*.csv',           # 비교할 run CSV들 (glob 패턴)
    'ds': 0.01,                           # 리샘플링 간격 (m)
    'out_dir': './out',                   # 출력 디렉토리
    'pass_cte95': 0.10,                   # CTE 95% 합격 기준 (m), None이면 체크 안함
    'pass_ate95': None,                   # ATE 95% 합격 기준 (m), None이면 체크 안함
    'pass_cte_rmse': None,                # CTE RMSE 합격 기준 (m), None이면 체크 안함
    'pass_ate_rmse': None,                # ATE RMSE 합격 기준 (m), None이면 체크 안함
}
# ============================================================================


def wrap_angle(theta: float) -> float:
    """각도를 [-pi, pi] 범위로 wrap"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def load_run_csv(filepath: str) -> Dict:
    """
    run CSV 로드
    
    Returns:
        {'t': np.array, 'x': np.array, 'y': np.array, 'yaw': np.array}
    """
    t, x, y, yaw = [], [], [], []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row['t']))
            x.append(float(row['x']))
            y.append(float(row['y']))
            yaw.append(float(row['yaw']))
    
    return {
        't': np.array(t),
        'x': np.array(x),
        'y': np.array(y),
        'yaw': np.array(yaw),
        'filepath': filepath
    }


def compute_arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """누적 arc-length 계산"""
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.zeros(len(x))
    s[1:] = np.cumsum(ds)
    return s


def resample_by_arc_length(run: Dict, ds: float) -> Dict:
    """
    arc-length 기준으로 리샘플링
    
    Args:
        run: {'t', 'x', 'y', 'yaw', ...}
        ds: 리샘플링 간격 (m)
    
    Returns:
        리샘플링된 run dict (+ 's' 추가)
    """
    s = compute_arc_length(run['x'], run['y'])
    
    if s[-1] < ds:
        # 너무 짧은 궤적
        return {
            's': s,
            't': run['t'],
            'x': run['x'],
            'y': run['y'],
            'yaw': run['yaw'],
            'filepath': run.get('filepath', '')
        }
    
    # 새 s grid
    s_new = np.arange(0, s[-1], ds)
    
    # 보간
    t_new = np.interp(s_new, s, run['t'])
    x_new = np.interp(s_new, s, run['x'])
    y_new = np.interp(s_new, s, run['y'])
    
    # yaw: unwrap -> 보간 -> wrap
    yaw_unwrapped = np.unwrap(run['yaw'])
    yaw_new = np.interp(s_new, s, yaw_unwrapped)
    yaw_new = np.array([wrap_angle(y) for y in yaw_new])
    
    return {
        's': s_new,
        't': t_new,
        'x': x_new,
        'y': y_new,
        'yaw': yaw_new,
        'filepath': run.get('filepath', '')
    }


def compute_ate(run: Dict, ref: Dict) -> Dict:
    """
    ATE (Absolute Trajectory Error) 계산
    
    동일 s에서의 위치 오차
    """
    # 공통 s 범위
    s_max = min(run['s'][-1], ref['s'][-1])
    mask_run = run['s'] <= s_max
    mask_ref = ref['s'] <= s_max
    
    # s 기준 보간
    s_common = run['s'][mask_run]
    
    x_run = run['x'][mask_run]
    y_run = run['y'][mask_run]
    
    x_ref = np.interp(s_common, ref['s'], ref['x'])
    y_ref = np.interp(s_common, ref['s'], ref['y'])
    
    # 오차
    dx = x_run - x_ref
    dy = y_run - y_ref
    e = np.sqrt(dx**2 + dy**2)
    
    return {
        's': s_common,
        'error': e,
        'mean': np.mean(e),
        'rmse': np.sqrt(np.mean(e**2)),
        'max': np.max(e),
        'p95': np.percentile(e, 95)
    }


def compute_cte(run: Dict, ref: Dict) -> Dict:
    """
    CTE (Cross-Track Error) 계산
    
    ref의 법선 방향 횡오차
    """
    # 공통 s 범위
    s_max = min(run['s'][-1], ref['s'][-1])
    mask_run = run['s'] <= s_max
    
    s_common = run['s'][mask_run]
    
    x_run = run['x'][mask_run]
    y_run = run['y'][mask_run]
    
    # ref 보간
    x_ref = np.interp(s_common, ref['s'], ref['x'])
    y_ref = np.interp(s_common, ref['s'], ref['y'])
    
    # ref heading (접선 방향) 계산
    yaw_ref_unwrapped = np.unwrap(ref['yaw'])
    yaw_ref = np.interp(s_common, ref['s'], yaw_ref_unwrapped)
    
    # 또는 수치 미분으로 계산
    # dx_ref = np.gradient(x_ref, s_common)
    # dy_ref = np.gradient(y_ref, s_common)
    # yaw_ref = np.arctan2(dy_ref, dx_ref)
    
    # normal 벡터 (법선: 접선을 90도 회전)
    normal_x = -np.sin(yaw_ref)
    normal_y = np.cos(yaw_ref)
    
    # 오차 벡터
    diff_x = x_run - x_ref
    diff_y = y_run - y_ref
    
    # CTE: diff와 normal의 내적 (부호 포함)
    cte = diff_x * normal_x + diff_y * normal_y
    cte_abs = np.abs(cte)
    
    return {
        's': s_common,
        'cte': cte,
        'cte_abs': cte_abs,
        'mean_abs': np.mean(cte_abs),
        'rmse': np.sqrt(np.mean(cte**2)),
        'max_abs': np.max(cte_abs),
        'p95_abs': np.percentile(cte_abs, 95)
    }


def plot_trajectories(runs: List[Dict], ref: Dict, out_path: str, title: str = "Trajectory Overlay"):
    """궤적 오버레이 플롯"""
    plt.figure(figsize=(12, 10))
    
    # ref (두껍게)
    plt.plot(ref['x'], ref['y'], 'k-', linewidth=2, label='Reference', zorder=10)
    
    # 시작점
    plt.plot(ref['x'][0], ref['y'][0], 'go', markersize=10, label='Start', zorder=11)
    
    # runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    for i, run in enumerate(runs):
        name = os.path.basename(run.get('filepath', f'run_{i}'))
        plt.plot(run['x'], run['y'], '-', color=colors[i], alpha=0.7, linewidth=1, label=name)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def plot_cte_vs_s(cte_results: List[Dict], run_names: List[str], out_path: str):
    """s-CTE 그래프"""
    plt.figure(figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(cte_results)))
    for i, (cte_res, name) in enumerate(zip(cte_results, run_names)):
        plt.plot(cte_res['s'], cte_res['cte'] * 100, '-', color=colors[i], alpha=0.7, label=name)
    
    plt.xlabel('Arc Length (m)')
    plt.ylabel('CTE (cm)')
    plt.title('Cross-Track Error vs Arc Length')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def plot_error_histogram(ate_results: List[Dict], out_path: str):
    """ATE 히스토그램"""
    plt.figure(figsize=(10, 6))
    
    all_errors = np.concatenate([r['error'] for r in ate_results])
    
    plt.hist(all_errors * 100, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('ATE (cm)')
    plt.ylabel('Count')
    plt.title(f'ATE Distribution (Mean: {np.mean(all_errors)*100:.2f} cm, 95%: {np.percentile(all_errors, 95)*100:.2f} cm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def plot_boxplot(ate_results: List[Dict], cte_results: List[Dict], run_names: List[str], out_path: str):
    """박스플롯"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ATE 박스플롯
    ate_data = [r['error'] * 100 for r in ate_results]  # cm
    axes[0].boxplot(ate_data, labels=[n.replace('.csv', '') for n in run_names])
    axes[0].set_ylabel('ATE (cm)')
    axes[0].set_title('Absolute Trajectory Error')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # CTE 박스플롯
    cte_data = [r['cte_abs'] * 100 for r in cte_results]  # cm
    axes[1].boxplot(cte_data, labels=[n.replace('.csv', '') for n in run_names])
    axes[1].set_ylabel('|CTE| (cm)')
    axes[1].set_title('Cross-Track Error (absolute)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='경로 반복성 평가')
    parser.add_argument('--ref', default=None, help='기준 run CSV 경로')
    parser.add_argument('--runs', default=None, nargs='+', help='비교할 run CSV들 (glob 지원)')
    parser.add_argument('--ds', type=float, default=None, help='리샘플링 간격 (m)')
    parser.add_argument('--out_dir', default=None, help='출력 디렉토리')
    parser.add_argument('--pass_cte95', type=float, default=None, help='CTE 95% 합격 기준 (m)')
    parser.add_argument('--pass_ate95', type=float, default=None, help='ATE 95% 합격 기준 (m)')
    parser.add_argument('--pass_cte_rmse', type=float, default=None, help='CTE RMSE 합격 기준 (m)')
    parser.add_argument('--pass_ate_rmse', type=float, default=None, help='ATE RMSE 합격 기준 (m)')
    
    args = parser.parse_args()
    
    # CLI 인자가 없으면 CONFIG 사용
    ref_path = args.ref if args.ref else CONFIG['ref']
    runs_pattern = args.runs if args.runs else [CONFIG['runs']]
    ds = args.ds if args.ds else CONFIG['ds']
    out_dir = args.out_dir if args.out_dir else CONFIG['out_dir']
    pass_cte95 = args.pass_cte95 if args.pass_cte95 else CONFIG['pass_cte95']
    pass_ate95 = args.pass_ate95 if args.pass_ate95 else CONFIG['pass_ate95']
    pass_cte_rmse = args.pass_cte_rmse if args.pass_cte_rmse else CONFIG['pass_cte_rmse']
    pass_ate_rmse = args.pass_ate_rmse if args.pass_ate_rmse else CONFIG['pass_ate_rmse']
    
    # 출력 디렉토리 생성
    os.makedirs(out_dir, exist_ok=True)
    
    # 기준 run 로드
    print(f"\n기준 run 로드: {ref_path}")
    ref_raw = load_run_csv(ref_path)
    ref = resample_by_arc_length(ref_raw, ds)
    print(f"  총 거리: {ref['s'][-1]:.2f} m")
    print(f"  리샘플 포인트: {len(ref['s'])}")
    
    # 비교 run들 로드
    run_files = []
    for pattern in runs_pattern:
        run_files.extend(glob(pattern))
    
    # ref 제외
    run_files = [f for f in run_files if os.path.abspath(f) != os.path.abspath(ref_path)]
    run_files = sorted(set(run_files))
    
    print(f"\n비교 run 수: {len(run_files)}")
    
    runs_raw = []
    runs_resampled = []
    for fp in run_files:
        print(f"  로드: {fp}")
        run_raw = load_run_csv(fp)
        runs_raw.append(run_raw)
        run_res = resample_by_arc_length(run_raw, ds)
        runs_resampled.append(run_res)
    
    # ATE/CTE 계산
    ate_results = []
    cte_results = []
    
    print(f"\n평가 결과:")
    print("="*80)
    
    for run, fp in zip(runs_resampled, run_files):
        name = os.path.basename(fp)
        ate = compute_ate(run, ref)
        cte = compute_cte(run, ref)
        
        ate_results.append(ate)
        cte_results.append(cte)
        
        print(f"\n{name}:")
        print(f"  ATE - Mean: {ate['mean']*100:.2f} cm, RMSE: {ate['rmse']*100:.2f} cm, "
              f"Max: {ate['max']*100:.2f} cm, 95%: {ate['p95']*100:.2f} cm")
        print(f"  CTE - Mean|: {cte['mean_abs']*100:.2f} cm, RMSE: {cte['rmse']*100:.2f} cm, "
              f"Max|: {cte['max_abs']*100:.2f} cm, 95%|: {cte['p95_abs']*100:.2f} cm")
    
    # 전체 통계
    print("\n" + "="*80)
    print("전체 통계:")
    
    all_ate = np.concatenate([r['error'] for r in ate_results])
    all_cte = np.concatenate([r['cte_abs'] for r in cte_results])
    
    summary = {
        'ref_file': ref_path,
        'run_files': run_files,
        'ds': ds,
        'total_distance_m': float(ref['s'][-1]),
        'n_runs': len(run_files),
        'ate': {
            'mean_cm': float(np.mean(all_ate) * 100),
            'rmse_cm': float(np.sqrt(np.mean(all_ate**2)) * 100),
            'max_cm': float(np.max(all_ate) * 100),
            'p95_cm': float(np.percentile(all_ate, 95) * 100)
        },
        'cte': {
            'mean_abs_cm': float(np.mean(all_cte) * 100),
            'rmse_cm': float(np.sqrt(np.mean(all_cte**2)) * 100),
            'max_abs_cm': float(np.max(all_cte) * 100),
            'p95_abs_cm': float(np.percentile(all_cte, 95) * 100)
        }
    }
    
    print(f"  ATE 전체 - Mean: {summary['ate']['mean_cm']:.2f} cm, 95%: {summary['ate']['p95_cm']:.2f} cm")
    print(f"  CTE 전체 - Mean|: {summary['cte']['mean_abs_cm']:.2f} cm, 95%|: {summary['cte']['p95_abs_cm']:.2f} cm")
    
    # 합격 판정
    passed = True
    print()
    
    if pass_cte95 is not None:
        cte95_m = np.percentile(all_cte, 95)
        if cte95_m <= pass_cte95:
            print(f"✓ CTE 95% 합격: {cte95_m*100:.2f} cm <= {pass_cte95*100:.2f} cm")
        else:
            print(f"✗ CTE 95% 불합격: {cte95_m*100:.2f} cm > {pass_cte95*100:.2f} cm")
            passed = False
    
    if pass_ate95 is not None:
        ate95_m = np.percentile(all_ate, 95)
        if ate95_m <= pass_ate95:
            print(f"✓ ATE 95% 합격: {ate95_m*100:.2f} cm <= {pass_ate95*100:.2f} cm")
        else:
            print(f"✗ ATE 95% 불합격: {ate95_m*100:.2f} cm > {pass_ate95*100:.2f} cm")
            passed = False
    
    if pass_cte_rmse is not None:
        cte_rmse_m = np.sqrt(np.mean(all_cte**2))
        if cte_rmse_m <= pass_cte_rmse:
            print(f"✓ CTE RMSE 합격: {cte_rmse_m*100:.2f} cm <= {pass_cte_rmse*100:.2f} cm")
        else:
            print(f"✗ CTE RMSE 불합격: {cte_rmse_m*100:.2f} cm > {pass_cte_rmse*100:.2f} cm")
            passed = False
    
    if pass_ate_rmse is not None:
        ate_rmse_m = np.sqrt(np.mean(all_ate**2))
        if ate_rmse_m <= pass_ate_rmse:
            print(f"✓ ATE RMSE 합격: {ate_rmse_m*100:.2f} cm <= {pass_ate_rmse*100:.2f} cm")
        else:
            print(f"✗ ATE RMSE 불합격: {ate_rmse_m*100:.2f} cm > {pass_ate_rmse*100:.2f} cm")
            passed = False
    
    summary['passed'] = passed
    
    # 결과 저장
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n요약 저장: {summary_path}")
    
    # 플롯 생성
    run_names = [os.path.basename(fp) for fp in run_files]
    
    # 궤적 오버레이
    plot_trajectories(
        runs_resampled, ref,
        os.path.join(out_dir, 'trajectory_overlay.png'),
        title=f'Trajectory Overlay (n={len(run_files)+1})'
    )
    
    # CTE vs s
    plot_cte_vs_s(
        cte_results, run_names,
        os.path.join(out_dir, 'cte_vs_arclength.png')
    )
    
    # ATE 히스토그램
    plot_error_histogram(
        ate_results,
        os.path.join(out_dir, 'ate_histogram.png')
    )
    
    # 박스플롯
    plot_boxplot(
        ate_results, cte_results, run_names,
        os.path.join(out_dir, 'error_boxplot.png')
    )
    
    print(f"\n완료! 결과: {out_dir}")
    
    return 0 if passed else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
