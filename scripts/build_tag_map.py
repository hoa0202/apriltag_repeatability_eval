#!/usr/bin/env python3
"""
edges.jsonl에서 tag_map.yaml 생성 (단방향 순차 방식)

Usage:
    python3 build_tag_map.py
    python3 build_tag_map.py --edges ./data/edges.jsonl --out ./config/tag_map.yaml
"""
import argparse
import json
import yaml
import numpy as np
from collections import defaultdict

# ============================================================================
# CONFIG: 기본 설정
# ============================================================================
CONFIG = {
    'edges': './data/edges.jsonl',
    'out': './config/tag_map.yaml',
    'ref_tag': 0,
    'visualize': True,
}
# ============================================================================


def compose_se2(pose, edge):
    """SE2 합성: pose ⊕ edge"""
    x, y, theta = pose
    dx, dy, dtheta = edge['dx'], edge['dy'], edge['dtheta']
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return (
        x + cos_t * dx - sin_t * dy,
        y + sin_t * dx + cos_t * dy,
        theta + dtheta
    )


def build_tag_map(edges_path, ref_tag=0):
    """
    edges.jsonl에서 tag_map 생성
    
    Args:
        edges_path: edges.jsonl 파일 경로
        ref_tag: 기준 태그 ID (원점에 배치)
    
    Returns:
        dict: {tag_id: (x, y, theta)}
    """
    # edges 로드
    edges = []
    with open(edges_path, 'r') as f:
        for line in f:
            edges.append(json.loads(line))
    
    print(f"Loaded {len(edges)} edges from {edges_path}")
    
    # 그래프 구성 (양방향)
    graph = defaultdict(dict)
    for e in edges:
        i, j = e['i'], e['j']
        graph[i][j] = e
        
        # 역방향 edge 계산
        inv_dx = -e['dx'] * np.cos(-e['dtheta']) + e['dy'] * np.sin(-e['dtheta'])
        inv_dy = -e['dx'] * np.sin(-e['dtheta']) - e['dy'] * np.cos(-e['dtheta'])
        graph[j][i] = {'dx': inv_dx, 'dy': inv_dy, 'dtheta': -e['dtheta']}
    
    # ref_tag부터 시작, 항상 더 큰 ID 방향으로만 이동
    # (한쪽 끝에서 시작하여 반대쪽으로 진행)
    poses = {ref_tag: (0, 0, 0)}
    visited = {ref_tag}
    stack = [ref_tag]
    
    while stack:
        current = stack.pop()
        for neighbor in sorted(graph[current].keys()):
            if neighbor > current and neighbor not in visited:
                edge = graph[current][neighbor]
                poses[neighbor] = compose_se2(poses[current], edge)
                visited.add(neighbor)
                stack.append(neighbor)
    
    print(f"Connected {len(poses)} tags (ID {min(poses.keys())} ~ {max(poses.keys())})")
    
    # 통계
    xs = [poses[i][0] for i in poses]
    ys = [poses[i][1] for i in poses]
    print(f"X range: {min(xs):.2f} ~ {max(xs):.2f}m")
    print(f"Y range: {min(ys):.2f} ~ {max(ys):.2f}m")
    
    return poses


def save_tag_map(poses, output_path, ref_tag=0):
    """tag_map.yaml 저장"""
    tag_map = {
        'reference_tag': ref_tag,
        'tags': {
            i: [float(poses[i][0]), float(poses[i][1]), float(poses[i][2])]
            for i in sorted(poses.keys())
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(tag_map, f, default_flow_style=False)
    
    print(f"Saved to {output_path}")


def visualize_tag_map(poses, output_path):
    """tag_map 시각화"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    xs = [poses[i][0] for i in sorted(poses.keys())]
    ys = [poses[i][1] for i in sorted(poses.keys())]
    ids = list(sorted(poses.keys()))
    
    plt.figure(figsize=(6, 14))
    plt.scatter(xs, ys, c=ids, cmap='viridis', s=20)
    plt.colorbar(label='Tag ID')
    
    # 시작/끝 표시
    plt.plot(poses[min(poses.keys())][0], poses[min(poses.keys())][1], 
             'go', markersize=12, markerfacecolor='none', markeredgewidth=2, label='Start')
    plt.plot(poses[max(poses.keys())][0], poses[max(poses.keys())][1],
             'ro', markersize=12, markerfacecolor='none', markeredgewidth=2, label='End')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('tag_map')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    vis_path = output_path.replace('.yaml', '_vis.png')
    plt.savefig(vis_path, dpi=150)
    print(f"Visualization saved to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Build tag_map.yaml from edges.jsonl')
    parser.add_argument('--edges', default=CONFIG['edges'], help='edges.jsonl path')
    parser.add_argument('--out', default=CONFIG['out'], help='output tag_map.yaml path')
    parser.add_argument('--ref', type=int, default=CONFIG['ref_tag'], help='reference tag ID')
    parser.add_argument('--no-vis', action='store_true', help='skip visualization')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Building tag_map from edges.jsonl")
    print("=" * 50)
    
    # tag_map 생성
    poses = build_tag_map(args.edges, args.ref)
    
    # 저장
    save_tag_map(poses, args.out, args.ref)
    
    # 시각화
    if not args.no_vis:
        visualize_tag_map(poses, args.out)
    
    print("=" * 50)
    print("Done!")


if __name__ == '__main__':
    main()
