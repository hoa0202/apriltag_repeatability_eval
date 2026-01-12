#!/usr/bin/env python3
"""
Phase B: tag0 기준 카메라 궤적 기록 노드

tag_map.yaml 기반으로 카메라 위치를 계산하고 CSV로 저장
RViz용 Path 토픽도 publish
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from apriltag_msgs.msg import AprilTagDetectionArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import yaml
import csv
import os
from typing import Dict, List, Tuple, Optional

from ..utils.se2 import SE2, se2_compose, se2_inv, se2_weighted_average, wrap_angle
from ..utils.quality import compute_tag_weight, QualityParams
from ..utils.tf_utils import TFHelper


class LocalizeAndRecordNode(Node):
    """Phase B: 카메라 궤적 기록 노드"""
    
    def __init__(self):
        super().__init__('localize_and_record')
        
        # 파라미터 선언
        self.declare_parameter('camera_frame', 'zed_left_camera_optical_frame')
        self.declare_parameter('tag_frame_prefix', 'tag_')
        self.declare_parameter('tag_map', './config/tag_map.yaml')
        self.declare_parameter('out_csv', 'auto')  # 'auto' = 자동 번호 부여
        self.declare_parameter('data_dir', './data')  # auto 모드용 디렉토리
        self.declare_parameter('detections_topic', '/detections')
        
        # 품질 파라미터
        self.declare_parameter('dm_min', 40.0)
        self.declare_parameter('dm_good', 70.0)
        self.declare_parameter('px_min', 70.0)
        self.declare_parameter('px_good', 140.0)
        self.declare_parameter('max_hamming', 0)
        
        # 디버그
        self.declare_parameter('debug', True)
        
        # Path publish 파라미터
        self.declare_parameter('publish_path', True)
        self.declare_parameter('path_frame', 'tag_0')  # Path 기준 프레임
        
        # Odom ground truth 파라미터
        self.declare_parameter('record_odom', True)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        
        # 파라미터 가져오기
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tag_frame_prefix = self.get_parameter('tag_frame_prefix').value
        self.tag_map_path = self.get_parameter('tag_map').value
        self.detections_topic = self.get_parameter('detections_topic').value
        
        # out_csv 처리 (auto면 자동 번호)
        out_csv_param = self.get_parameter('out_csv').value
        data_dir = self.get_parameter('data_dir').value
        
        if out_csv_param == 'auto':
            self.out_csv_path = self._get_next_run_path(data_dir)
        else:
            self.out_csv_path = out_csv_param
        
        self.quality_params = QualityParams(
            dm_min=self.get_parameter('dm_min').value,
            dm_good=self.get_parameter('dm_good').value,
            px_min=self.get_parameter('px_min').value,
            px_good=self.get_parameter('px_good').value,
            max_hamming=self.get_parameter('max_hamming').value
        )
        
        self.debug = self.get_parameter('debug').value
        self.publish_path = self.get_parameter('publish_path').value
        self.path_frame = self.get_parameter('path_frame').value
        
        # Odom 설정
        self.record_odom = self.get_parameter('record_odom').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        
        # Odom 기준점 (첫 프레임에서 설정)
        self.odom_origin: Optional[SE2] = None
        self.apriltag_origin: Optional[SE2] = None
        
        # TF 헬퍼
        self.tf_helper = TFHelper(self)
        
        # 태그 맵 로드
        self.tag_map: Dict[int, SE2] = {}
        self._load_tag_map()
        
        # 출력 CSV 준비
        self._prepare_csv()
        
        # 통계
        self.frame_count = 0
        self.record_count = 0
        self.tags_per_frame: List[int] = []
        
        # Path 메시지
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.path_frame
        
        # QoS 설정
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 구독
        self.sub_detections = self.create_subscription(
            AprilTagDetectionArray,
            self.detections_topic,
            self.detection_callback,
            qos
        )
        
        # Path publisher
        if self.publish_path:
            self.pub_path = self.create_publisher(Path, '~/camera_path', 10)
        
        self.get_logger().info(f"LocalizeAndRecord 노드 시작")
        self.get_logger().info(f"  Camera frame: {self.camera_frame}")
        self.get_logger().info(f"  Tag map: {self.tag_map_path}")
        self.get_logger().info(f"  Output CSV: {self.out_csv_path}")
        self.get_logger().info(f"  태그 맵에 {len(self.tag_map)}개 태그 로드됨")
    
    def _get_next_run_path(self, data_dir: str) -> str:
        """다음 run 번호 자동 계산"""
        import re
        import glob
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        
        # 기존 run_*.csv 파일들 찾기
        pattern = os.path.join(data_dir, 'run_*.csv')
        existing = glob.glob(pattern)
        
        # 번호 추출
        numbers = []
        for f in existing:
            match = re.search(r'run_(\d+)\.csv$', f)
            if match:
                numbers.append(int(match.group(1)))
        
        # 다음 번호
        next_num = max(numbers) + 1 if numbers else 1
        
        return os.path.join(data_dir, f'run_{next_num:02d}.csv')
    
    def _load_tag_map(self):
        """tag_map.yaml 로드"""
        with open(self.tag_map_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.ref_tag_id = data.get('reference_tag', 0)
        
        for tag_id, pose in data['tags'].items():
            self.tag_map[int(tag_id)] = SE2(pose[0], pose[1], pose[2])
        
        self.get_logger().info(f"태그 맵 로드: {len(self.tag_map)}개 태그, ref={self.ref_tag_id}")
    
    def _prepare_csv(self):
        """CSV 파일 준비"""
        out_dir = os.path.dirname(self.out_csv_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # CSV 헤더 작성
        with open(self.out_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if self.record_odom:
                writer.writerow(['t', 'x', 'y', 'yaw', 'odom_x', 'odom_y', 'odom_yaw', 
                               'error_x', 'error_y', 'error_dist', 'tags_used', 'quality'])
            else:
            writer.writerow(['t', 'x', 'y', 'yaw', 'tags_used', 'quality'])
    
    def detection_callback(self, msg: AprilTagDetectionArray):
        """detection 메시지 콜백"""
        stamp = Time.from_msg(msg.header.stamp)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.frame_count += 1
        
        # 1) 각 태그에 대해 TF lookup + 품질 계산 + 카메라 포즈 후보 계산
        pose_candidates: List[Tuple[SE2, float, int]] = []  # (pose, weight, tag_id)
        
        for det in msg.detections:
            tag_id = det.id
            
            # 태그 맵에 없는 태그는 skip
            if tag_id not in self.tag_map:
                continue
            
            # 코너 추출
            corners = [(c.x, c.y) for c in det.corners]
            
            # 품질 계산
            weight = compute_tag_weight(
                decision_margin=det.decision_margin,
                corners=corners,
                hamming=det.hamming,
                params=self.quality_params
            )
            
            if weight <= 0:
                continue
            
            # TF lookup
            result = self.tf_helper.get_tag_pose_se2(
                camera_frame=self.camera_frame,
                tag_id=tag_id,
                stamp=stamp,
                tag_frame_prefix=self.tag_frame_prefix
            )
            
            if result is None:
                continue
            
            T_cam_tag, _ = result  # T_cam_tag (SE2)
            
            # 태그 맵에서 tag0->tagk 변환
            T_tag0_tagk = self.tag_map[tag_id]
            
            # 카메라 포즈 계산: T_tag0_cam = T_tag0_tagk ⊕ inv(T_cam_tagk)
            T_cam_tag_inv = se2_inv(T_cam_tag)
            T_tag0_cam = se2_compose(T_tag0_tagk, T_cam_tag_inv)
            
            pose_candidates.append((T_tag0_cam, weight, tag_id))
        
        # 통계
        self.tags_per_frame.append(len(pose_candidates))
        
        if len(pose_candidates) == 0:
            return
        
        # 2) 멀티 태그 fusion
        poses = [p[0] for p in pose_candidates]
        weights = [p[1] for p in pose_candidates]
        tags_used = [str(p[2]) for p in pose_candidates]
        
        fused_pose = se2_weighted_average(poses, weights)
        avg_quality = sum(weights) / len(weights)
        
        # 3) Odom lookup (ground truth)
        odom_pose = None
        error_info = None
        if self.record_odom:
            odom_result = self.tf_helper.lookup_se2(self.odom_frame, self.base_frame, stamp)
            if odom_result is not None:
                odom_pose = odom_result
                
                # 첫 프레임: 기준점 설정
                if self.odom_origin is None:
                    self.odom_origin = odom_pose
                    self.apriltag_origin = fused_pose
                
                # 상대 위치 계산
                from ..utils.se2 import se2_between
                rel_apriltag = se2_between(self.apriltag_origin, fused_pose)
                rel_odom = se2_between(self.odom_origin, odom_pose)
                
                # 좌표계 변환 (camera_link -> base_link)
                scale = 1.0 / 1.84
                rel_apriltag_x = -rel_apriltag.y * scale
                rel_apriltag_y = rel_apriltag.x * scale
                
                # 오차 계산
                import math
                error_x = rel_apriltag_x - rel_odom.x
                error_y = rel_apriltag_y - rel_odom.y
                error_dist = math.sqrt(error_x**2 + error_y**2)
                error_info = (rel_odom.x, rel_odom.y, rel_odom.theta, error_x, error_y, error_dist)
        
        # 4) CSV 기록
        self._write_record(timestamp, fused_pose, tags_used, avg_quality, error_info)
        self.record_count += 1
        
        # 5) Path publish
        if self.publish_path:
            self._add_to_path(msg.header.stamp, fused_pose)
        
        # 디버그 출력
        if self.debug and self.frame_count % 30 == 0:
            avg_tags = sum(self.tags_per_frame[-100:]) / min(len(self.tags_per_frame), 100)
            fail_rate = self.tf_helper.get_fail_rate() * 100
            
            error_str = ""
            if error_info is not None:
                error_str = f", Error: {error_info[5]*100:.1f}cm"
            
            self.get_logger().info(
                f"[Frame {self.frame_count}] "
                f"Pos: ({fused_pose.x:.3f}, {fused_pose.y:.3f}), "
                f"Tags: {len(pose_candidates)}, "
                f"Records: {self.record_count}{error_str}"
            )
    
    def _write_record(self, timestamp: float, pose: SE2, tags_used: List[str], quality: float, 
                      error_info: Optional[tuple] = None):
        """CSV에 기록 추가"""
        with open(self.out_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if self.record_odom and error_info is not None:
                odom_x, odom_y, odom_yaw, error_x, error_y, error_dist = error_info
                writer.writerow([
                    f"{timestamp:.6f}",
                    f"{pose.x:.6f}",
                    f"{pose.y:.6f}",
                    f"{pose.theta:.6f}",
                    f"{odom_x:.6f}",
                    f"{odom_y:.6f}",
                    f"{odom_yaw:.6f}",
                    f"{error_x:.6f}",
                    f"{error_y:.6f}",
                    f"{error_dist:.6f}",
                    "|".join(tags_used),
                    f"{quality:.4f}"
                ])
            else:
            writer.writerow([
                f"{timestamp:.6f}",
                f"{pose.x:.6f}",
                f"{pose.y:.6f}",
                f"{pose.theta:.6f}",
                "|".join(tags_used),
                f"{quality:.4f}"
            ])
    
    def _add_to_path(self, stamp, pose: SE2):
        """Path 메시지에 포즈 추가 + publish"""
        import math
        
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = self.path_frame
        
        pose_stamped.pose.position.x = pose.x
        pose_stamped.pose.position.y = pose.y
        pose_stamped.pose.position.z = 0.0
        
        # yaw -> quaternion (z축 회전)
        pose_stamped.pose.orientation.x = 0.0
        pose_stamped.pose.orientation.y = 0.0
        pose_stamped.pose.orientation.z = math.sin(pose.theta / 2)
        pose_stamped.pose.orientation.w = math.cos(pose.theta / 2)
        
        self.path_msg.poses.append(pose_stamped)
        self.path_msg.header.stamp = stamp
        
        self.pub_path.publish(self.path_msg)
    
    def destroy_node(self):
        """노드 종료 시 통계 출력"""
        self.get_logger().info("="*50)
        self.get_logger().info("LocalizeAndRecord 종료 통계:")
        self.get_logger().info(f"  총 프레임: {self.frame_count}")
        self.get_logger().info(f"  기록된 포즈: {self.record_count}")
        if self.tags_per_frame:
            self.get_logger().info(f"  평균 태그/프레임: {sum(self.tags_per_frame)/len(self.tags_per_frame):.2f}")
        self.get_logger().info(f"  TF 실패율: {self.tf_helper.get_fail_rate()*100:.1f}%")
        self.get_logger().info(f"  출력 CSV: {self.out_csv_path}")
        self.get_logger().info("="*50)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocalizeAndRecordNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
