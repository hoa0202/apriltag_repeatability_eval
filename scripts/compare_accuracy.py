#!/usr/bin/env python3
"""
정확도 분석 스크립트: AprilTag vs Ground Truth 비교

Usage:
    python3 compare_accuracy.py
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ============================================================================
# CONFIG: 여기서 파라미터 수정
# ============================================================================
CONFIG = {
    'input_csv': './data/accuracy_test.csv',   # accuracy_test 노드 출력
    'out_dir': './out/accuracy',               # 출력 디렉토리
    'auto_align': False,                       # 이미 노드에서 좌표 변환함
    'align_samples': 100,                      # 정렬에 사용할 초기 샘플 수
}
# ============================================================================


def procrustes_align(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> tuple:
    """
    Procrustes analysis: source를 target에 맞추는 최적 스케일+회전+평행이동 찾기
    
    Args:
        source: (N, 2) 배열 - 변환할 좌표
        target: (N, 2) 배열 - 목표 좌표
        with_scale: 스케일 보정 포함 여부
    
    Returns:
        (R, t, scale, aligned, angle): 회전행렬, 평행이동, 스케일, 정렬된 source, 회전각도
    """
    # 중심 이동
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    
    source_centered = source - source_mean
    target_centered = target - target_mean
    
    # SVD로 최적 회전 찾기
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # reflection 방지
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 스케일 계산
    if with_scale:
        source_var = np.sum(source_centered ** 2)
        scale = np.sum(S) / source_var if source_var > 0 else 1.0
    else:
        scale = 1.0
    
    # 평행이동
    t = target_mean - scale * R @ source_mean
    
    # 정렬된 좌표
    aligned = scale * (R @ source.T).T + t
    
    # 회전 각도 추출
    angle = np.arctan2(R[1, 0], R[0, 0])
    
    return R, t, scale, aligned, angle


def apply_alignment(x: np.ndarray, y: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> tuple:
    """정렬 변환 적용 (스케일 + 회전 + 평행이동)"""
    coords = np.column_stack([x, y])
    aligned = scale * (R @ coords.T).T + t
    return aligned[:, 0], aligned[:, 1]


def load_accuracy_csv(filepath: str) -> dict:
    """accuracy_test.csv 로드"""
    data = {
        't': [], 
        'error_x': [], 'error_y': [], 'error_dist': [], 'error_yaw': [],
        'rel_apriltag_x': [], 'rel_apriltag_y': [],
        'rel_odom_x': [], 'rel_odom_y': [],
    }
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['t'].append(float(row['t']))
            data['error_x'].append(float(row['error_x']))
            data['error_y'].append(float(row['error_y']))
            data['error_dist'].append(float(row['error_dist']))
            data['error_yaw'].append(float(row['error_yaw']))
            data['rel_apriltag_x'].append(float(row['rel_apriltag_x']))
            data['rel_apriltag_y'].append(float(row['rel_apriltag_y']))
            data['rel_odom_x'].append(float(row['rel_odom_x']))
            data['rel_odom_y'].append(float(row['rel_odom_y']))
    
    # numpy 변환
    for k in data:
        data[k] = np.array(data[k])
    
    return data


def compute_statistics(errors: np.ndarray) -> dict:
    """오차 통계 계산"""
    return {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'max': float(np.max(errors)),
        'min': float(np.min(errors)),
        'p50': float(np.percentile(errors, 50)),
        'p95': float(np.percentile(errors, 95)),
        'p99': float(np.percentile(errors, 99)),
    }


def plot_trajectory_comparison(data: dict, out_path: str):
    """궤적 비교 플롯"""
    plt.figure(figsize=(10, 10))
    
    plt.plot(data['rel_odom_x'], data['rel_odom_y'], 'b-', 
             linewidth=2, label='Ground Truth (Odom)', alpha=0.7)
    
    # 정렬된 좌표가 있으면 사용
    if 'aligned_apriltag_x' in data:
        plt.plot(data['aligned_apriltag_x'], data['aligned_apriltag_y'], 'r-', 
                 linewidth=2, label='AprilTag (aligned)', alpha=0.7)
    else:
        plt.plot(data['rel_apriltag_x'], data['rel_apriltag_y'], 'r-', 
                 linewidth=2, label='AprilTag', alpha=0.7)
    
    # 시작점
    plt.plot(0, 0, 'go', markersize=12, label='Start', zorder=10)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Comparison: AprilTag vs Ground Truth')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def plot_error_over_time(data: dict, out_path: str):
    """시간에 따른 오차 플롯"""
    t = data['t'] - data['t'][0]  # 상대 시간
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 위치 오차
    axes[0].plot(t, data['error_dist'] * 100, 'b-', linewidth=1)
    axes[0].set_ylabel('Position Error (cm)')
    axes[0].set_title('Position Error Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # X, Y 오차
    axes[1].plot(t, data['error_x'] * 100, 'r-', linewidth=1, label='X error')
    axes[1].plot(t, data['error_y'] * 100, 'g-', linewidth=1, label='Y error')
    axes[1].set_ylabel('Error (cm)')
    axes[1].set_title('X/Y Error Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Yaw 오차
    axes[2].plot(t, np.degrees(data['error_yaw']), 'purple', linewidth=1)
    axes[2].set_ylabel('Yaw Error (deg)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Yaw Error Over Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def plot_error_histogram(data: dict, out_path: str):
    """오차 히스토그램"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 위치 오차 히스토그램
    errors_cm = data['error_dist'] * 100
    axes[0].hist(errors_cm, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(errors_cm), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(errors_cm):.2f} cm')
    axes[0].axvline(np.percentile(errors_cm, 95), color='orange', linestyle='--',
                    label=f'95%: {np.percentile(errors_cm, 95):.2f} cm')
    axes[0].set_xlabel('Position Error (cm)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Position Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Yaw 오차 히스토그램
    yaw_errors_deg = np.degrees(np.abs(data['error_yaw']))
    axes[1].hist(yaw_errors_deg, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1].axvline(np.mean(yaw_errors_deg), color='r', linestyle='--',
                    label=f'Mean: {np.mean(yaw_errors_deg):.2f}°')
    axes[1].axvline(np.percentile(yaw_errors_deg, 95), color='orange', linestyle='--',
                    label=f'95%: {np.percentile(yaw_errors_deg, 95):.2f}°')
    axes[1].set_xlabel('|Yaw Error| (deg)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Yaw Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def plot_error_scatter(data: dict, out_path: str):
    """X-Y 오차 scatter plot"""
    plt.figure(figsize=(8, 8))
    
    errors_x_cm = data['error_x'] * 100
    errors_y_cm = data['error_y'] * 100
    
    plt.scatter(errors_x_cm, errors_y_cm, alpha=0.3, s=10)
    
    # 원 그리기 (RMSE, 95%)
    rmse = np.sqrt(np.mean(data['error_dist']**2)) * 100
    p95 = np.percentile(data['error_dist'], 95) * 100
    
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(rmse * np.cos(theta), rmse * np.sin(theta), 'r--', 
             linewidth=2, label=f'RMSE: {rmse:.2f} cm')
    plt.plot(p95 * np.cos(theta), p95 * np.sin(theta), 'orange', 
             linewidth=2, linestyle='--', label=f'95%: {p95:.2f} cm')
    
    plt.xlabel('X Error (cm)')
    plt.ylabel('Y Error (cm)')
    plt.title('Error Distribution (X-Y)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"플롯 저장: {out_path}")


def main():
    input_csv = CONFIG['input_csv']
    out_dir = CONFIG['out_dir']
    auto_align = CONFIG.get('auto_align', False)
    align_samples = CONFIG.get('align_samples', 100)
    
    # 출력 디렉토리 생성
    os.makedirs(out_dir, exist_ok=True)
    
    # 데이터 로드
    print(f"\n데이터 로드: {input_csv}")
    data = load_accuracy_csv(input_csv)
    print(f"  샘플 수: {len(data['t'])}")
    
    # 자동 좌표계 정렬
    if auto_align:
        print(f"\n좌표계 자동 정렬 (Procrustes + Scale, 처음 {align_samples}개 샘플 사용)")
        
        n = min(align_samples, len(data['t']))
        source = np.column_stack([data['rel_apriltag_x'][:n], data['rel_apriltag_y'][:n]])
        target = np.column_stack([data['rel_odom_x'][:n], data['rel_odom_y'][:n]])
        
        R, t, scale, _, angle = procrustes_align(source, target, with_scale=True)
        print(f"  스케일: {scale:.4f} (AprilTag 좌표에 곱함)")
        print(f"  회전 각도: {np.degrees(angle):.2f}°")
        print(f"  평행이동: ({t[0]*100:.2f}, {t[1]*100:.2f}) cm")
        
        # 전체 데이터에 정렬 적용
        aligned_x, aligned_y = apply_alignment(
            data['rel_apriltag_x'], data['rel_apriltag_y'], R, t, scale
        )
        
        # 오차 재계산
        data['error_x'] = aligned_x - data['rel_odom_x']
        data['error_y'] = aligned_y - data['rel_odom_y']
        data['error_dist'] = np.sqrt(data['error_x']**2 + data['error_y']**2)
        
        # 플롯용으로 정렬된 좌표 저장
        data['aligned_apriltag_x'] = aligned_x
        data['aligned_apriltag_y'] = aligned_y
    
    # 통계 계산
    print("\n" + "="*60)
    print("정확도 분석 결과" + (" (정렬 후)" if auto_align else ""))
    print("="*60)
    
    pos_stats = compute_statistics(data['error_dist'])
    yaw_stats = compute_statistics(np.abs(data['error_yaw']))
    
    print(f"\n위치 오차 (Position Error):")
    print(f"  Mean:  {pos_stats['mean']*100:.2f} cm")
    print(f"  Std:   {pos_stats['std']*100:.2f} cm")
    print(f"  RMSE:  {pos_stats['rmse']*100:.2f} cm")
    print(f"  Max:   {pos_stats['max']*100:.2f} cm")
    print(f"  50%:   {pos_stats['p50']*100:.2f} cm")
    print(f"  95%:   {pos_stats['p95']*100:.2f} cm")
    print(f"  99%:   {pos_stats['p99']*100:.2f} cm")
    
    print(f"\nYaw 오차 (Yaw Error):")
    print(f"  Mean:  {np.degrees(yaw_stats['mean']):.2f}°")
    print(f"  Std:   {np.degrees(yaw_stats['std']):.2f}°")
    print(f"  RMSE:  {np.degrees(yaw_stats['rmse']):.2f}°")
    print(f"  Max:   {np.degrees(yaw_stats['max']):.2f}°")
    print(f"  95%:   {np.degrees(yaw_stats['p95']):.2f}°")
    
    # 요약 저장
    summary = {
        'input_file': input_csv,
        'n_samples': len(data['t']),
        'duration_sec': float(data['t'][-1] - data['t'][0]),
        'position_error': {
            'mean_cm': pos_stats['mean'] * 100,
            'std_cm': pos_stats['std'] * 100,
            'rmse_cm': pos_stats['rmse'] * 100,
            'max_cm': pos_stats['max'] * 100,
            'p50_cm': pos_stats['p50'] * 100,
            'p95_cm': pos_stats['p95'] * 100,
            'p99_cm': pos_stats['p99'] * 100,
        },
        'yaw_error': {
            'mean_deg': np.degrees(yaw_stats['mean']),
            'std_deg': np.degrees(yaw_stats['std']),
            'rmse_deg': np.degrees(yaw_stats['rmse']),
            'max_deg': np.degrees(yaw_stats['max']),
            'p95_deg': np.degrees(yaw_stats['p95']),
        }
    }
    
    summary_path = os.path.join(out_dir, 'accuracy_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n요약 저장: {summary_path}")
    
    # 플롯 생성
    print("\n플롯 생성 중...")
    
    plot_trajectory_comparison(data, os.path.join(out_dir, 'trajectory_comparison.png'))
    plot_error_over_time(data, os.path.join(out_dir, 'error_over_time.png'))
    plot_error_histogram(data, os.path.join(out_dir, 'error_histogram.png'))
    plot_error_scatter(data, os.path.join(out_dir, 'error_scatter.png'))
    
    print(f"\n완료! 결과: {out_dir}")


if __name__ == '__main__':
    main()
