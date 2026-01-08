"""TF lookup 헬퍼 모듈"""
import rclpy
from rclpy.time import Time, Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
from typing import Optional, Tuple
import numpy as np

from .se2 import SE2, project_to_se2


class TFHelper:
    """TF lookup 헬퍼 클래스"""
    
    def __init__(self, node: Node, buffer_duration: float = 10.0):
        """
        Args:
            node: ROS2 노드
            buffer_duration: TF 버퍼 유지 시간 (초)
        """
        self.node = node
        self.tf_buffer = Buffer(cache_time=Duration(seconds=buffer_duration))
        self.tf_listener = TransformListener(self.tf_buffer, node)
        
        # 통계
        self.lookup_count = 0
        self.fail_count = 0
    
    def lookup_transform(self, 
                         target_frame: str, 
                         source_frame: str,
                         stamp: Time,
                         timeout_sec: float = 0.1) -> Optional[TransformStamped]:
        """
        TF lookup 수행
        
        Args:
            target_frame: 기준 프레임 (예: camera frame)
            source_frame: 대상 프레임 (예: tag frame)
            stamp: 시간
            timeout_sec: 타임아웃 (초)
        
        Returns:
            TransformStamped or None (실패 시)
        """
        self.lookup_count += 1
        
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=timeout_sec)
            )
            return transform
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.fail_count += 1
            self.node.get_logger().debug(f"TF lookup failed: {target_frame} -> {source_frame}: {e}")
            return None
    
    def lookup_transform_latest(self,
                                target_frame: str,
                                source_frame: str) -> Optional[TransformStamped]:
        """
        최신 TF lookup (fallback용)
        """
        self.lookup_count += 1
        
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time()  # Time(0) = latest
            )
            return transform
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.fail_count += 1
            self.node.get_logger().debug(f"TF lookup (latest) failed: {target_frame} -> {source_frame}: {e}")
            return None
    
    def get_tag_pose_se2(self,
                         camera_frame: str,
                         tag_id: int,
                         stamp: Time,
                         tag_frame_prefix: str = "tag_") -> Optional[Tuple[SE2, TransformStamped]]:
        """
        카메라 기준 태그의 SE2 포즈 획득
        
        Args:
            camera_frame: 카메라 프레임 이름
            tag_id: 태그 ID
            stamp: 시간
            tag_frame_prefix: 태그 프레임 접두사 (기본 "tag_")
        
        Returns:
            (SE2 포즈, 원본 TransformStamped) or None
        """
        tag_frame = f"{tag_frame_prefix}{tag_id}"
        
        # exact time 시도, 실패시 latest로 fallback
        transform = self.lookup_transform(camera_frame, tag_frame, stamp)
        if transform is None:
            transform = self.lookup_transform_latest(camera_frame, tag_frame)
        if transform is None:
            return None
        
        # TransformStamped -> SE2
        trans = transform.transform.translation
        rot = transform.transform.rotation
        
        se2_pose = project_to_se2(
            (trans.x, trans.y, trans.z),
            (rot.x, rot.y, rot.z, rot.w)
        )
        
        return se2_pose, transform
    
    def get_fail_rate(self) -> float:
        """TF lookup 실패율 반환"""
        if self.lookup_count == 0:
            return 0.0
        return self.fail_count / self.lookup_count
    
    def reset_stats(self):
        """통계 리셋"""
        self.lookup_count = 0
        self.fail_count = 0


def transform_to_se2(transform: TransformStamped) -> SE2:
    """TransformStamped를 SE2로 변환"""
    trans = transform.transform.translation
    rot = transform.transform.rotation
    
    return project_to_se2(
        (trans.x, trans.y, trans.z),
        (rot.x, rot.y, rot.z, rot.w)
    )


def invert_transform_se2(transform: TransformStamped) -> SE2:
    """TransformStamped의 역변환을 SE2로 반환"""
    from .se2 import se2_inv
    se2 = transform_to_se2(transform)
    return se2_inv(se2)
