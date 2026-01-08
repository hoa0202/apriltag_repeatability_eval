"""SE2 수학 유틸리티 모듈"""
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class SE2:
    """SE2 포즈 (x, y, theta)"""
    x: float
    y: float
    theta: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SE2':
        return cls(arr[0], arr[1], arr[2])
    
    def __repr__(self):
        return f"SE2(x={self.x:.4f}, y={self.y:.4f}, theta={self.theta:.4f})"


def wrap_angle(theta: float) -> float:
    """각도를 [-pi, pi] 범위로 wrap"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """연속적인 각도 배열을 unwrap (점프 제거)"""
    return np.unwrap(angles)


def se2_compose(a: SE2, b: SE2) -> SE2:
    """SE2 합성: a ⊕ b"""
    cos_a = np.cos(a.theta)
    sin_a = np.sin(a.theta)
    
    x = a.x + cos_a * b.x - sin_a * b.y
    y = a.y + sin_a * b.x + cos_a * b.y
    theta = wrap_angle(a.theta + b.theta)
    
    return SE2(x, y, theta)


def se2_inv(a: SE2) -> SE2:
    """SE2 역변환: inv(a)"""
    cos_a = np.cos(a.theta)
    sin_a = np.sin(a.theta)
    
    x = -cos_a * a.x - sin_a * a.y
    y = sin_a * a.x - cos_a * a.y
    theta = wrap_angle(-a.theta)
    
    return SE2(x, y, theta)


def se2_between(a: SE2, b: SE2) -> SE2:
    """두 SE2 사이의 상대변환: inv(a) ⊕ b"""
    return se2_compose(se2_inv(a), b)


def project_to_se2(translation: Tuple[float, float, float], 
                   quaternion: Tuple[float, float, float, float]) -> SE2:
    """
    3D pose를 SE2로 투영
    
    Args:
        translation: (x, y, z)
        quaternion: (qx, qy, qz, qw)
    
    Returns:
        SE2 포즈 (천장 카메라 가정: z축이 위, xy 평면이 바닥)
    """
    x, y, z = translation
    qx, qy, qz, qw = quaternion
    
    # quaternion -> yaw (z축 회전만 추출)
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return SE2(x, y, wrap_angle(yaw))


def se2_weighted_average(poses: List[SE2], weights: List[float]) -> SE2:
    """
    SE2 포즈들의 가중 평균
    
    x, y: 가중 평균
    theta: 원형 평균 (circular mean)
    """
    if len(poses) == 0:
        raise ValueError("Empty poses list")
    
    if len(poses) == 1:
        return poses[0]
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # normalize
    
    # x, y 가중 평균
    x_avg = sum(w * p.x for w, p in zip(weights, poses))
    y_avg = sum(w * p.y for w, p in zip(weights, poses))
    
    # theta 원형 평균
    sin_sum = sum(w * np.sin(p.theta) for w, p in zip(weights, poses))
    cos_sum = sum(w * np.cos(p.theta) for w, p in zip(weights, poses))
    theta_avg = np.arctan2(sin_sum, cos_sum)
    
    return SE2(x_avg, y_avg, wrap_angle(theta_avg))


def se2_to_matrix(pose: SE2) -> np.ndarray:
    """SE2를 3x3 변환행렬로 변환"""
    c = np.cos(pose.theta)
    s = np.sin(pose.theta)
    return np.array([
        [c, -s, pose.x],
        [s,  c, pose.y],
        [0,  0,     1]
    ])


def matrix_to_se2(mat: np.ndarray) -> SE2:
    """3x3 변환행렬을 SE2로 변환"""
    x = mat[0, 2]
    y = mat[1, 2]
    theta = np.arctan2(mat[1, 0], mat[0, 0])
    return SE2(x, y, wrap_angle(theta))


def interpolate_se2(p0: SE2, p1: SE2, t: float) -> SE2:
    """두 SE2 사이를 선형 보간 (t: 0~1)"""
    x = p0.x + t * (p1.x - p0.x)
    y = p0.y + t * (p1.y - p0.y)
    
    # 각도는 최단 경로로 보간
    dtheta = wrap_angle(p1.theta - p0.theta)
    theta = wrap_angle(p0.theta + t * dtheta)
    
    return SE2(x, y, theta)
