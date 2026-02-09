#!/usr/bin/env python3
"""
여러 run CSV 비교 시각화 스크립트

Usage:
    python3 visualize_run.py                    # run_01 기준으로 모든 run 비교
    python3 visualize_run.py --data-dir ./data  # 데이터 디렉토리 지정
    python3 visualize_run.py --ref run_01.csv   # 레퍼런스 run 지정
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

# ============================================================================
# CONFIG: 기본 설정
# ============================================================================
CONFIG = {
    'data_dir': './data',
    'ref': 'run_01.csv',
    'out_dir': './data',
}
# ============================================================================


def load_run_csv(csv_path, rotate_odom=True):
    """run CSV 로드"""
    t, x, y, yaw = [], [], [], []
    odom_x, odom_y = [], []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) >= 12:  # odom 있는 행
                t.append(float(row[0]))
                x.append(float(row[1]))
                y.append(float(row[2]))
                yaw.append(float(row[3]))
                odom_x.append(float(row[4]))
                odom_y.append(float(row[5]))
            elif len(row) >= 4:
                t.append(float(row[0]))
                x.append(float(row[1]))
                y.append(float(row[2]))
                yaw.append(float(row[3]))
    
    odom_x, odom_y = np.array(odom_x), np.array(odom_y)
    
    # Odom 90도 회전 (AprilTag 좌표계 맞춤)
    if rotate_odom and len(odom_x) > 0:
        odom_x_rot = odom_y.copy()
        odom_y_rot = -odom_x.copy()
    else:
        odom_x_rot, odom_y_rot = odom_x, odom_y
    
    return {
        't': np.array(t),
        'x': np.array(x),
        'y': np.array(y),
        'yaw': np.array(yaw),
        'odom_x': odom_x_rot,
        'odom_y': odom_y_rot,
    }


def resample_by_distance(x, y, ds=0.01):
    """거리 기반 리샘플링"""
    if len(x) < 2:
        return x, y
    
    # 누적 거리 계산
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    cumsum = np.concatenate([[0], np.cumsum(dist)])
    
    # 등간격 리샘플링
    total_dist = cumsum[-1]
    if total_dist < ds:
        return x, y
    
    new_s = np.arange(0, total_dist, ds)
    new_x = np.interp(new_s, cumsum, x)
    new_y = np.interp(new_s, cumsum, y)
    
    return new_x, new_y


def compute_cte(ref_x, ref_y, test_x, test_y):
    """Cross Track Error 계산"""
    errors = []
    for tx, ty in zip(test_x, test_y):
        dists = np.sqrt((ref_x - tx)**2 + (ref_y - ty)**2)
        errors.append(np.min(dists))
    return np.array(errors)


def visualize_runs(data_dir='./data', ref_name='run_01.csv', out_dir='./data'):
    """여러 run 비교 시각화 (AprilTag + Odom)"""
    
    # run 파일들 찾기
    pattern = os.path.join(data_dir, 'run_*.csv')
    run_files = sorted(glob(pattern))
    
    if not run_files:
        print(f"No run files found in {data_dir}")
        return
    
    print("=" * 60)
    print(f"Run 파일 비교 ({len(run_files)}개)")
    print("=" * 60)
    
    # 레퍼런스 로드
    ref_path = os.path.join(data_dir, ref_name)
    if not os.path.exists(ref_path):
        ref_path = run_files[0]
    
    ref_data = load_run_csv(ref_path)
    ref_x, ref_y = resample_by_distance(ref_data['x'], ref_data['y'])
    has_odom = len(ref_data['odom_x']) > 0
    if has_odom:
        ref_odom_x, ref_odom_y = resample_by_distance(ref_data['odom_x'], ref_data['odom_y'])
    
    print(f"레퍼런스: {os.path.basename(ref_path)} ({len(ref_data['x'])} 샘플, odom: {len(ref_data['odom_x'])})")
    
    # 모든 run 로드
    runs = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_files)))
    
    for i, f in enumerate(run_files):
        name = os.path.basename(f)
        data = load_run_csv(f)
        runs[name] = {
            'data': data,
            'color': colors[i],
            'x_rs': resample_by_distance(data['x'], data['y'])[0],
            'y_rs': resample_by_distance(data['x'], data['y'])[1],
        }
        if len(data['odom_x']) > 0:
            runs[name]['odom_x_rs'] = resample_by_distance(data['odom_x'], data['odom_y'])[0]
            runs[name]['odom_y_rs'] = resample_by_distance(data['odom_x'], data['odom_y'])[1]
        print(f"  {name}: {len(data['x'])} 샘플, odom: {len(data['odom_x'])}")
    
    # CTE 계산
    print(f"\n=== AprilTag Cross Track Error (CTE) ===")
    cte_results = {}
    for name, run in runs.items():
        if name == os.path.basename(ref_path):
            continue
        cte = compute_cte(ref_x, ref_y, run['x_rs'], run['y_rs'])
        cte_results[name] = cte
        print(f"  {name}: mean={cte.mean()*100:.2f}cm, max={cte.max()*100:.2f}cm, 95%={np.percentile(cte, 95)*100:.2f}cm")
    
    # Odom CTE 계산
    odom_cte_results = {}
    if has_odom:
        print(f"\n=== Odom Cross Track Error (CTE) ===")
        for name, run in runs.items():
            if name == os.path.basename(ref_path):
                continue
            if 'odom_x_rs' in run:
                cte = compute_cte(ref_odom_x, ref_odom_y, run['odom_x_rs'], run['odom_y_rs'])
                odom_cte_results[name] = cte
                print(f"  {name}: mean={cte.mean()*100:.2f}cm, max={cte.max()*100:.2f}cm, 95%={np.percentile(cte, 95)*100:.2f}cm")
    
    # 플롯 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. AprilTag 경로 비교
    ax1 = axes[0, 0]
    for name, run in runs.items():
        is_ref = name == os.path.basename(ref_path)
        ax1.plot(run['data']['x'], run['data']['y'], 
                 color=run['color'], 
                 linewidth=2 if is_ref else 1,
                 linestyle='-' if is_ref else '--',
                 alpha=0.9 if is_ref else 0.7,
                 label=f"{name.replace('.csv','')} {'(ref)' if is_ref else ''}")
    
    ax1.plot(ref_data['x'][0], ref_data['y'][0], 'go', markersize=10, label='Start')
    ax1.plot(ref_data['x'][-1], ref_data['y'][-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'AprilTag Trajectory ({len(runs)} runs)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Odom 경로 비교
    ax2 = axes[0, 1]
    if has_odom:
        for name, run in runs.items():
            if len(run['data']['odom_x']) == 0:
                continue
            is_ref = name == os.path.basename(ref_path)
            ax2.plot(run['data']['odom_x'], run['data']['odom_y'], 
                     color=run['color'], 
                     linewidth=2 if is_ref else 1,
                     linestyle='-' if is_ref else '--',
                     alpha=0.9 if is_ref else 0.7,
                     label=f"{name.replace('.csv','')} {'(ref)' if is_ref else ''}")
        
        ax2.plot(ref_data['odom_x'][0], ref_data['odom_y'][0], 'go', markersize=10)
        ax2.plot(ref_data['odom_x'][-1], ref_data['odom_y'][-1], 'ro', markersize=10)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Odom Trajectory (90° rotated)')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
    else:
        ax2.text(0.5, 0.5, 'No Odom data', ha='center', va='center', fontsize=14)
        ax2.set_title('Odom Trajectory')
    
    # 3. AprilTag CTE 비교
    ax3 = axes[1, 0]
    if cte_results:
        cte_data = [v * 100 for v in cte_results.values()]
        cte_labels = list(cte_results.keys())
        
        bp = ax3.boxplot(cte_data, labels=[l.replace('.csv', '') for l in cte_labels], 
                         patch_artist=True)
        for patch, name in zip(bp['boxes'], cte_labels):
            patch.set_facecolor(runs[name]['color'])
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('CTE (cm)')
        ax3.set_title('AprilTag CTE Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        means = [np.mean(d) for d in cte_data]
        ax3.scatter(range(1, len(means)+1), means, color='red', s=50, zorder=5, label='Mean')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Only 1 run\n(no comparison)', ha='center', va='center', fontsize=14)
        ax3.set_title('AprilTag CTE')
    
    # 4. Odom CTE 비교
    ax4 = axes[1, 1]
    if odom_cte_results:
        cte_data = [v * 100 for v in odom_cte_results.values()]
        cte_labels = list(odom_cte_results.keys())
        
        bp = ax4.boxplot(cte_data, labels=[l.replace('.csv', '') for l in cte_labels], 
                         patch_artist=True)
        for patch, name in zip(bp['boxes'], cte_labels):
            patch.set_facecolor(runs[name]['color'])
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('CTE (cm)')
        ax4.set_title('Odom CTE Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
        means = [np.mean(d) for d in cte_data]
        ax4.scatter(range(1, len(means)+1), means, color='red', s=50, zorder=5, label='Mean')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Odom CTE', ha='center', va='center', fontsize=14)
        ax4.set_title('Odom CTE')
    
    plt.tight_layout()
    
    # 저장
    out_path = os.path.join(out_dir, 'runs_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n저장됨: {out_path}")
    
    # 통계 요약
    print(f"\n=== 요약 ===")
    if cte_results:
        all_cte = np.concatenate(list(cte_results.values()))
        print(f"AprilTag CTE: mean={all_cte.mean()*100:.2f}cm, 95%={np.percentile(all_cte, 95)*100:.2f}cm")
    if odom_cte_results:
        all_odom_cte = np.concatenate(list(odom_cte_results.values()))
        print(f"Odom CTE: mean={all_odom_cte.mean()*100:.2f}cm, 95%={np.percentile(all_odom_cte, 95)*100:.2f}cm")
    
    return runs


def main():
    parser = argparse.ArgumentParser(description='Compare multiple run CSVs')
    parser.add_argument('--data-dir', default=CONFIG['data_dir'], help='Data directory with run_*.csv files')
    parser.add_argument('--ref', default=CONFIG['ref'], help='Reference run filename')
    parser.add_argument('--out-dir', default=CONFIG['out_dir'], help='Output directory')
    
    args = parser.parse_args()
    
    visualize_runs(args.data_dir, args.ref, args.out_dir)


if __name__ == '__main__':
    main()
