"""태그 품질/가중치 계산 모듈"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QualityParams:
    """품질 계산 파라미터"""
    # decision_margin 기준
    dm_min: float = 40.0      # 이 이하면 weight=0
    dm_good: float = 70.0     # 이 이상이면 weight=1
    
    # pixel size 기준 (corners 기반 평균 변 길이)
    px_min: float = 70.0      # 이 이하면 weight=0
    px_good: float = 140.0    # 이 이상이면 weight=1
    
    # hamming 기준
    max_hamming: int = 0      # 이보다 크면 drop


def clamp(value: float, min_val: float, max_val: float) -> float:
    """값을 [min_val, max_val] 범위로 클램프"""
    return max(min_val, min(max_val, value))


def compute_pixel_size(corners: List[Tuple[float, float]]) -> float:
    """
    corners로부터 평균 변 길이(px) 계산
    
    Args:
        corners: [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] 4개 코너
    
    Returns:
        평균 변 길이 (pixels)
    """
    if len(corners) != 4:
        return 0.0
    
    corners = np.array(corners)
    
    # 4개 변의 길이 계산
    edge_lengths = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        edge_lengths.append(np.linalg.norm(p1 - p0))
    
    return np.mean(edge_lengths)


def compute_tag_weight(decision_margin: float,
                       corners: List[Tuple[float, float]],
                       hamming: int,
                       params: Optional[QualityParams] = None) -> float:
    """
    단일 태그의 품질 가중치 계산
    
    Args:
        decision_margin: apriltag detection의 decision_margin
        corners: 4개 코너 좌표 [(x,y), ...]
        hamming: hamming distance (0이 best)
        params: 품질 파라미터 (None이면 기본값)
    
    Returns:
        weight in [0, 1]
    """
    if params is None:
        params = QualityParams()
    
    # hamming 체크
    if hamming > params.max_hamming:
        return 0.0
    
    # decision_margin 기반 가중치
    if decision_margin < params.dm_min:
        w_dm = 0.0
    elif decision_margin >= params.dm_good:
        w_dm = 1.0
    else:
        w_dm = (decision_margin - params.dm_min) / (params.dm_good - params.dm_min)
    
    # pixel size 기반 가중치
    px_size = compute_pixel_size(corners)
    if px_size < params.px_min:
        w_px = 0.0
    elif px_size >= params.px_good:
        w_px = 1.0
    else:
        w_px = (px_size - params.px_min) / (params.px_good - params.px_min)
    
    # 최종 가중치 (곱)
    return w_dm * w_px


def compute_edge_weight(w_i: float, w_j: float) -> float:
    """
    엣지 가중치 계산 (두 태그의 품질 곱)
    
    Args:
        w_i: 태그 i의 가중치
        w_j: 태그 j의 가중치
    
    Returns:
        엣지 가중치
    """
    return w_i * w_j


def is_valid_edge(dx: float, dy: float, dtheta: float,
                  max_dist: float = 5.0,
                  max_angle: float = np.pi / 4) -> bool:
    """
    엣지가 유효한지 검사 (이상치 제거)
    
    Args:
        dx, dy: 상대 변위 (m)
        dtheta: 상대 각도 (rad)
        max_dist: 최대 허용 거리 (m)
        max_angle: 최대 허용 각도 차이 (rad)
    
    Returns:
        유효 여부
    """
    dist = np.sqrt(dx * dx + dy * dy)
    if dist > max_dist:
        return False
    
    if abs(dtheta) > max_angle:
        return False
    
    return True
