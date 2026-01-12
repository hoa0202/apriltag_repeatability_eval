#!/usr/bin/env python3
"""
Phase A 오프라인 최적화: Pose Graph 최적화로 tag_map.yaml 생성

Usage:
    # CLI 모드
    python3 solve_pose_graph.py --edges ./data/edges.jsonl --out ./config/tag_map.yaml
    
    # 직접 실행 모드 (아래 CONFIG 섹션 수정 후)
    python3 solve_pose_graph.py
"""
import argparse
import json
import yaml
import numpy as np
from scipy.optimize import least_squares
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys
import os

# ============================================================================
# CONFIG: 여기서 파라미터 수정 (CLI 인자 없이 실행할 때 사용됨)
# ============================================================================
CONFIG = {
    'edges': './data/edges.jsonl',        # edges.jsonl 파일 경로
    'out': './config/tag_map.yaml',       # 출력 tag_map.yaml 경로
    'k_theta': 0.5,                       # 각도 스케일 계수
    'ref_tag': 0,                         # 기준 태그 ID (없으면 최소 ID로 자동 대체)
    'quiet': False,                       # True면 출력 최소화
}
# ============================================================================


def wrap_angle(theta: float) -> float:
    """각도를 [-pi, pi] 범위로 wrap"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def se2_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SE2 합성: a ⊕ b"""
    cos_a = np.cos(a[2])
    sin_a = np.sin(a[2])
    
    x = a[0] + cos_a * b[0] - sin_a * b[1]
    y = a[1] + sin_a * b[0] + cos_a * b[1]
    theta = wrap_angle(a[2] + b[2])
    
    return np.array([x, y, theta])


def se2_inv(a: np.ndarray) -> np.ndarray:
    """SE2 역변환: inv(a)"""
    cos_a = np.cos(a[2])
    sin_a = np.sin(a[2])
    
    x = -cos_a * a[0] - sin_a * a[1]
    y = sin_a * a[0] - cos_a * a[1]
    theta = wrap_angle(-a[2])
    
    return np.array([x, y, theta])


def se2_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 SE2 사이의 상대변환: inv(a) ⊕ b"""
    return se2_compose(se2_inv(a), b)


class PoseGraphSolver:
    """33노드 Pose Graph 최적화 솔버"""
    
    def __init__(self, k_theta: float = 0.5):
        """
        Args:
            k_theta: 각도 스케일 (rad -> m 변환 계수)
        """
        self.k_theta = k_theta
        self.edges: List[Dict] = []
        self.tag_ids: set = set()
        
        # 참조 태그 (원점으로 고정)
        self.ref_tag_id = 0
    
    def load_edges(self, filepath: str) -> int:
        """edges.jsonl 파일 로드"""
        self.edges = []
        self.tag_ids = set()
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                edge = json.loads(line)
                self.edges.append(edge)
                self.tag_ids.add(edge['i'])
                self.tag_ids.add(edge['j'])
        
        print(f"로드된 엣지: {len(self.edges)}")
        print(f"발견된 태그 ID: {sorted(self.tag_ids)}")
        print(f"총 태그 수: {len(self.tag_ids)}")
        
        return len(self.edges)
    
    def _get_tag_index(self, tag_id: int) -> Optional[int]:
        """태그 ID -> 변수 인덱스 (ref_tag는 None)"""
        if tag_id == self.ref_tag_id:
            return None
        return self.sorted_ids.index(tag_id)
    
    def _get_pose(self, tag_id: int, params: np.ndarray) -> np.ndarray:
        """태그 ID로부터 포즈 획득"""
        if tag_id == self.ref_tag_id:
            return np.array([0.0, 0.0, 0.0])
        idx = self._get_tag_index(tag_id)
        return params[idx*3:(idx+1)*3]
    
    def _residual(self, params: np.ndarray) -> np.ndarray:
        """잔차 벡터 계산"""
        residuals = []
        
        for edge in self.edges:
            i, j = edge['i'], edge['j']
            z = np.array([edge['dx'], edge['dy'], edge['dtheta']])
            w = edge['w']
            
            # 추정 포즈
            Xi = self._get_pose(i, params)
            Xj = self._get_pose(j, params)
            
            # 예측 상대변환
            pred = se2_between(Xi, Xj)
            
            # 오차: inv(Z) ⊕ pred
            err = se2_between(z, pred)
            
            # weighted residual
            sqrt_w = np.sqrt(w)
            residuals.extend([
                sqrt_w * err[0],
                sqrt_w * err[1],
                sqrt_w * self.k_theta * wrap_angle(err[2])
            ])
        
        return np.array(residuals)
    
    def _initialize_bfs(self) -> np.ndarray:
        """BFS로 초기값 구성"""
        # 태그별 엣지 수집
        edges_by_tag: Dict[int, List[Dict]] = defaultdict(list)
        for edge in self.edges:
            edges_by_tag[edge['i']].append(edge)
            edges_by_tag[edge['j']].append(edge)
        
        # ref_tag 제외한 태그들
        self.sorted_ids = sorted([tid for tid in self.tag_ids if tid != self.ref_tag_id])
        n_tags = len(self.sorted_ids)
        
        # 초기화: 모두 0
        init = np.zeros(n_tags * 3)
        initialized = {self.ref_tag_id}
        
        # BFS 큐
        queue = [self.ref_tag_id]
        
        while queue and len(initialized) < len(self.tag_ids):
            current = queue.pop(0)
            current_pose = self._get_pose(current, init) if current != self.ref_tag_id else np.array([0.0, 0.0, 0.0])
            
            for edge in edges_by_tag[current]:
                i, j = edge['i'], edge['j']
                other = j if i == current else i
                
                if other in initialized:
                    continue
                
                # 상대변환으로 초기 포즈 계산
                z = np.array([edge['dx'], edge['dy'], edge['dtheta']])
                
                if i == current:
                    # current -> other: other = current ⊕ z
                    other_pose = se2_compose(current_pose, z)
                else:
                    # other -> current: other = current ⊕ inv(z)
                    other_pose = se2_compose(current_pose, se2_inv(z))
                
                # 저장
                idx = self._get_tag_index(other)
                if idx is not None:
                    init[idx*3:(idx+1)*3] = other_pose
                
                initialized.add(other)
                queue.append(other)
        
        not_init = self.tag_ids - initialized
        if not_init:
            print(f"WARNING: 다음 태그들이 그래프에서 분리됨: {not_init}")
        
        return init
    
    def solve(self, verbose: bool = True) -> Dict[int, List[float]]:
        """최적화 수행"""
        if not self.edges:
            raise ValueError("엣지가 없습니다. load_edges()를 먼저 호출하세요.")
        
        # ref_tag가 데이터에 있는지 확인
        if self.ref_tag_id not in self.tag_ids:
            print(f"WARNING: ref_tag {self.ref_tag_id}가 데이터에 없음. 최소 ID로 대체.")
            self.ref_tag_id = min(self.tag_ids)
        
        # 변수 구성
        self.sorted_ids = sorted([tid for tid in self.tag_ids if tid != self.ref_tag_id])
        n_vars = len(self.sorted_ids) * 3
        
        if verbose:
            print(f"\n최적화 시작:")
            print(f"  변수 수: {n_vars} (태그 {len(self.sorted_ids)}개 × 3)")
            print(f"  엣지 수: {len(self.edges)}")
            print(f"  기준 태그: {self.ref_tag_id}")
        
        # 초기값
        x0 = self._initialize_bfs()
        
        # 초기 잔차 통계
        res0 = self._residual(x0)
        if verbose:
            print(f"\n초기 잔차:")
            print(f"  Mean: {np.mean(np.abs(res0)):.6f}")
            print(f"  RMSE: {np.sqrt(np.mean(res0**2)):.6f}")
            print(f"  Max:  {np.max(np.abs(res0)):.6f}")
        
        # 최적화
        result = least_squares(
            self._residual,
            x0,
            loss='huber',
            f_scale=0.05,
            verbose=2 if verbose else 0,
            max_nfev=5000
        )
        
        # 최종 잔차 통계
        res_final = self._residual(result.x)
        if verbose:
            print(f"\n최종 잔차:")
            print(f"  Mean: {np.mean(np.abs(res_final)):.6f}")
            print(f"  RMSE: {np.sqrt(np.mean(res_final**2)):.6f}")
            print(f"  Max:  {np.max(np.abs(res_final)):.6f}")
            print(f"  95%:  {np.percentile(np.abs(res_final), 95):.6f}")
        
        # 결과 정리
        tag_map = {self.ref_tag_id: [0.0, 0.0, 0.0]}
        
        for tag_id in self.sorted_ids:
            idx = self._get_tag_index(tag_id)
            pose = result.x[idx*3:(idx+1)*3]
            tag_map[tag_id] = [
                round(float(pose[0]), 6),
                round(float(pose[1]), 6),
                round(float(wrap_angle(pose[2])), 6)
            ]
        
        if verbose:
            print(f"\n추정된 태그 위치:")
            for tag_id in sorted(tag_map.keys()):
                pose = tag_map[tag_id]
                print(f"  tag_{tag_id}: x={pose[0]:.4f}, y={pose[1]:.4f}, θ={np.degrees(pose[2]):.2f}°")
        
        return tag_map
    
    def save_map(self, tag_map: Dict[int, List[float]], filepath: str):
        """tag_map을 yaml로 저장"""
        output = {
            'reference_tag': self.ref_tag_id,
            'tags': {int(k): v for k, v in tag_map.items()}
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n태그 맵 저장: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Pose Graph 최적화로 tag_map.yaml 생성')
    parser.add_argument('--edges', default=None, help='edges.jsonl 파일 경로')
    parser.add_argument('--out', default=None, help='출력 tag_map.yaml 경로')
    parser.add_argument('--k_theta', type=float, default=None, help='각도 스케일 계수')
    parser.add_argument('--ref_tag', type=int, default=None, help='기준 태그 ID')
    parser.add_argument('--quiet', action='store_true', help='출력 최소화')
    
    args = parser.parse_args()
    
    # CLI 인자가 없으면 CONFIG 사용
    edges_path = args.edges if args.edges else CONFIG['edges']
    out_path = args.out if args.out else CONFIG['out']
    k_theta = args.k_theta if args.k_theta else CONFIG['k_theta']
    ref_tag = args.ref_tag if args.ref_tag is not None else CONFIG['ref_tag']
    quiet = args.quiet or CONFIG['quiet']
    
    solver = PoseGraphSolver(k_theta=k_theta)
    solver.ref_tag_id = ref_tag
    
    # 엣지 로드
    n_edges = solver.load_edges(edges_path)
    
    if n_edges == 0:
        print("ERROR: 엣지가 없습니다.")
        sys.exit(1)
    
    # 최적화
    tag_map = solver.solve(verbose=not quiet)
    
    # 저장
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    solver.save_map(tag_map, out_path)


if __name__ == '__main__':
    main()
