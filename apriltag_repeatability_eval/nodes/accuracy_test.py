#!/usr/bin/env python3
"""
정확도 테스트 노드: AprilTag Localization vs Ground Truth (TF odom->base_link) 비교

AprilTag로 계산한 위치와 시뮬레이터 Ground Truth를 동시에 기록
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from apriltag_msgs.msg import AprilTagDetectionArray
import yaml
import csv
import os
from typing import Dict, List, Tuple, Optional
import math

from ..utils.se2 import SE2, se2_compose, se2_inv, se2_weighted_average, se2_between, wrap_angle
from ..utils.quality import compute_tag_weight, QualityParams
from ..utils.tf_utils import TFHelper


class AccuracyTestNode(Node):
    """AprilTag vs Ground Truth 비교 노드"""
    
    def __init__(self):
        super().__init__('accuracy_test')
        
        # 파라미터 선언
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('tag_frame_prefix', 'tag_')
        self.declare_parameter('tag_map', './config/tag_map.yaml')
        self.declare_parameter('out_csv', './data/accuracy_test.csv')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_link_frame', 'base_link')
        
        # 품질 파라미터
        self.declare_parameter('dm_min', 40.0)
        self.declare_parameter('dm_good', 70.0)
        self.declare_parameter('px_min', 70.0)
        self.declare_parameter('px_good', 140.0)
        self.declare_parameter('max_hamming', 0)
        
        # 디버그
        self.declare_parameter('debug', True)
        
        # 파라미터 가져오기
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tag_frame_prefix = self.get_parameter('tag_frame_prefix').value
        self.tag_map_path = self.get_parameter('tag_map').value
        self.out_csv_path = self.get_parameter('out_csv').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_link_frame = self.get_parameter('base_link_frame').value
        
        self.quality_params = QualityParams(
            dm_min=self.get_parameter('dm_min').value,
            dm_good=self.get_parameter('dm_good').value,
            px_min=self.get_parameter('px_min').value,
            px_good=self.get_parameter('px_good').value,
            max_hamming=self.get_parameter('max_hamming').value
        )
        
        self.debug = self.get_parameter('debug').value
        
        # TF 헬퍼
        self.tf_helper = TFHelper(self)
        
        # 태그 맵 로드
        self.tag_map: Dict[int, SE2] = {}
        self._load_tag_map()
        
        # 출력 CSV 준비
        self._prepare_csv()
        
        # 첫 프레임 기준점 (정렬용)
        self.apriltag_origin: Optional[SE2] = None
        self.odom_origin: Optional[SE2] = None
        
        # 통계
        self.record_count = 0
        self.frame_count = 0
        
        # QoS 설정
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 구독 (AprilTag만)
        self.sub_detections = self.create_subscription(
            AprilTagDetectionArray,
            self.detections_topic,
            self.detection_callback,
            qos_sensor
        )
        
        self.get_logger().info(f"AccuracyTest 노드 시작")
        self.get_logger().info(f"  Camera frame: {self.camera_frame}")
        self.get_logger().info(f"  Tag map: {self.tag_map_path}")
        self.get_logger().info(f"  Output CSV: {self.out_csv_path}")
        self.get_logger().info(f"  Ground Truth TF: {self.odom_frame} -> {self.base_link_frame}")
    
    def _load_tag_map(self):
        """tag_map.yaml 로드"""
        with open(self.tag_map_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.ref_tag_id = data.get('reference_tag', 0)
        
        for tag_id, pose in data['tags'].items():
            self.tag_map[int(tag_id)] = SE2(pose[0], pose[1], pose[2])
        
        self.get_logger().info(f"태그 맵 로드: {len(self.tag_map)}개 태그, ref={self.ref_tag_id}")
    
    def _prepare_csv(self):
        """CSV 파일 준비 (기존 파일 삭제 후 새로 생성)"""
        out_dir = os.path.dirname(self.out_csv_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # 기존 파일 삭제
        if os.path.exists(self.out_csv_path):
            os.remove(self.out_csv_path)
            self.get_logger().info(f"기존 CSV 삭제: {self.out_csv_path}")
        
        # CSV 헤더 작성
        with open(self.out_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                't',
                'apriltag_x', 'apriltag_y', 'apriltag_yaw',
                'odom_x', 'odom_y', 'odom_yaw',
                'rel_apriltag_x', 'rel_apriltag_y', 'rel_apriltag_yaw',
                'rel_odom_x', 'rel_odom_y', 'rel_odom_yaw',
                'error_x', 'error_y', 'error_dist', 'error_yaw',
                'tags_used', 'quality'
            ])
    
    def _get_ground_truth_pose(self) -> Optional[SE2]:
        """TF에서 odom -> base_link 변환을 가져와 SE2로 변환"""
        transform = self.tf_helper.lookup_transform_latest(
            self.odom_frame, self.base_link_frame
        )
        if transform is None:
            return None
        
        # Transform -> SE2 (변환 없이 원본 사용)
        pos = transform.transform.translation
        ori = transform.transform.rotation
        
        # quaternion -> yaw
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return SE2(pos.x, pos.y, yaw)
    
    def detection_callback(self, msg: AprilTagDetectionArray):
        """detection 메시지 콜백"""
        stamp = Time.from_msg(msg.header.stamp)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.frame_count += 1
        
        # Ground Truth TF lookup
        odom_pose = self._get_ground_truth_pose()
        if odom_pose is None:
            if self.debug and self.frame_count % 30 == 0:
                self.get_logger().warn(f"[Frame {self.frame_count}] TF {self.odom_frame}->{self.base_link_frame} 없음")
            return
        
        # 디버그: detection 개수
        if self.debug and self.frame_count % 30 == 0:
            self.get_logger().info(f"[Frame {self.frame_count}] Detections: {len(msg.detections)}")
        
        # AprilTag localization
        pose_candidates: List[Tuple[SE2, float, int]] = []
        
        tf_fail_count = 0
        not_in_map_count = 0
        weight_zero_count = 0
        
        for det in msg.detections:
            tag_id = det.id
            
            if tag_id not in self.tag_map:
                not_in_map_count += 1
                continue
            
            corners = [(c.x, c.y) for c in det.corners]
            
            weight = compute_tag_weight(
                decision_margin=det.decision_margin,
                corners=corners,
                hamming=det.hamming,
                params=self.quality_params
            )
            
            if weight <= 0:
                weight_zero_count += 1
                continue
            
            result = self.tf_helper.get_tag_pose_se2(
                camera_frame=self.camera_frame,
                tag_id=tag_id,
                stamp=stamp,
                tag_frame_prefix=self.tag_frame_prefix
            )
            
            if result is None:
                tf_fail_count += 1
                continue
            
            T_cam_tag, _ = result
            T_tag0_tagk = self.tag_map[tag_id]
            T_cam_tag_inv = se2_inv(T_cam_tag)
            T_tag0_cam = se2_compose(T_tag0_tagk, T_cam_tag_inv)
            
            pose_candidates.append((T_tag0_cam, weight, tag_id))
        
        if len(pose_candidates) == 0:
            if self.debug and self.frame_count % 30 == 0:
                self.get_logger().warn(
                    f"[Frame {self.frame_count}] pose_candidates=0, "
                    f"not_in_map={not_in_map_count}, weight_zero={weight_zero_count}, tf_fail={tf_fail_count}"
                )
            return
        
        # Fusion
        poses = [p[0] for p in pose_candidates]
        weights = [p[1] for p in pose_candidates]
        tags_used = [str(p[2]) for p in pose_candidates]
        
        apriltag_pose = se2_weighted_average(poses, weights)
        avg_quality = sum(weights) / len(weights)
        
        # 각 태그별 개별 오차 출력 (첫 프레임 이후, 30프레임마다)
        if self.debug and self.apriltag_origin is not None and self.frame_count % 90 == 0:
            rel_odom_dbg = se2_between(self.odom_origin, odom_pose)
            self.get_logger().info(f"[Frame {self.frame_count}] 태그별 오차:")
            for pose, weight, tag_id in pose_candidates:
                rel_tag = se2_between(self.apriltag_origin, pose)
                err = math.sqrt((rel_tag.x - rel_odom_dbg.x)**2 + (rel_tag.y - rel_odom_dbg.y)**2)
                self.get_logger().info(f"  tag_{tag_id}: ({rel_tag.x*100:6.1f}, {rel_tag.y*100:6.1f}) cm, err={err*100:.1f}cm")
        
        # 첫 프레임: 기준점 설정
        if self.apriltag_origin is None:
            self.apriltag_origin = apriltag_pose
            self.odom_origin = odom_pose
            self.get_logger().info(f"기준점 설정됨")
        
        # 상대 위치 계산 (첫 프레임 대비)
        rel_apriltag = se2_between(self.apriltag_origin, apriltag_pose)
        rel_odom = se2_between(self.odom_origin, odom_pose)
        
        # camera_link → base_link 좌표계 변환
        # AprilTag Y → ROS X (부호 반전), AprilTag X → ROS Y
        # 스케일 보정: TF가 ~1.84배 크게 측정됨
        scale = 1.0 / 1.84
        rel_apriltag_x = -rel_apriltag.y * scale
        rel_apriltag_y = rel_apriltag.x * scale
        
        # 오차 계산
        error_x = rel_apriltag_x - rel_odom.x
        error_y = rel_apriltag_y - rel_odom.y
        error_dist = math.sqrt(error_x**2 + error_y**2)
        error_yaw = wrap_angle(rel_apriltag.theta - rel_odom.theta)
        
        # CSV 기록 (변환된 좌표 사용)
        self._write_record(
            timestamp,
            apriltag_pose, odom_pose,
            rel_apriltag_x, rel_apriltag_y, rel_apriltag.theta,
            rel_odom,
            error_x, error_y, error_dist, error_yaw,
            tags_used, avg_quality
        )
        self.record_count += 1
        
        # 디버그 출력 (좌표 상세 + 사용된 태그)
        if self.debug and self.frame_count % 30 == 0:
            tags_str = ','.join(tags_used)
            self.get_logger().info(
                f"[Frame {self.frame_count}] Tags:[{tags_str}] "
                f"Err: {error_dist*100:.1f} cm | "
                f"AprilTag: ({rel_apriltag_x*100:7.1f}, {rel_apriltag_y*100:7.1f}) cm | "
                f"Odom: ({rel_odom.x*100:7.1f}, {rel_odom.y*100:7.1f}) cm"
            )
    
    def _write_record(self, timestamp, apriltag_pose, odom_pose,
                      rel_apriltag_x, rel_apriltag_y, rel_apriltag_yaw,
                      rel_odom,
                      error_x, error_y, error_dist, error_yaw,
                      tags_used, quality):
        """CSV에 기록"""
        with open(self.out_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{timestamp:.6f}",
                f"{apriltag_pose.x:.6f}", f"{apriltag_pose.y:.6f}", f"{apriltag_pose.theta:.6f}",
                f"{odom_pose.x:.6f}", f"{odom_pose.y:.6f}", f"{odom_pose.theta:.6f}",
                f"{rel_apriltag_x:.6f}", f"{rel_apriltag_y:.6f}", f"{rel_apriltag_yaw:.6f}",
                f"{rel_odom.x:.6f}", f"{rel_odom.y:.6f}", f"{rel_odom.theta:.6f}",
                f"{error_x:.6f}", f"{error_y:.6f}", f"{error_dist:.6f}", f"{error_yaw:.6f}",
                "|".join(tags_used),
                f"{quality:.4f}"
            ])
    
    def destroy_node(self):
        """노드 종료 시 통계 출력"""
        self.get_logger().info("="*50)
        self.get_logger().info("AccuracyTest 종료 통계:")
        self.get_logger().info(f"  총 프레임: {self.frame_count}")
        self.get_logger().info(f"  기록된 샘플: {self.record_count}")
        self.get_logger().info(f"  출력 CSV: {self.out_csv_path}")
        self.get_logger().info("="*50)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AccuracyTestNode()
    
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
