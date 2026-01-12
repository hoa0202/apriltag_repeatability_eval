#!/usr/bin/env python3
"""
Phase A: 태그-태그 엣지 수집 노드 (고정밀 버전)

특징:
  - 정지 상태에서만 수집 (Odom 속도 체크)
  - 표준편차 수렴 기반 충분성 판단
  - Outlier rejection (IQR)
  - Median 사용 (noise-robust)
  - 실시간 품질 피드백

사용법:
  ros2 launch apriltag_repeatability_eval phase_a.launch.py

시나리오:
  1. 노드 실행
  2. 로봇 이동 → 구역 도착 → 3~5초 정지 (자동 수집)
  3. 터미널에서 각 edge std 확인
  4. 모든 edge가 "✓" 되면 Ctrl+C → edges.jsonl 저장
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from apriltag_msgs.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry
import json
import os
import math
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ..utils.se2 import SE2, se2_between, wrap_angle
from ..utils.quality import compute_tag_weight, compute_edge_weight, is_valid_edge, QualityParams
from ..utils.tf_utils import TFHelper, transform_to_se2


class CollectEdgesNode(Node):
    """Phase A: 태그-태그 엣지 수집 노드 (고정밀 버전)"""
    
    # === 설정 ===
    CONFIG = {
        # 품질 기준
        'velocity_threshold': 0.005,   # 정지 판정 속도 (m/s)
        'min_samples': 30,             # edge당 최소 샘플 수
        'target_std_mm': 1.0,          # 목표 표준편차 (mm)
        'outlier_iqr_factor': 1.5,     # IQR outlier 제거 계수
    }
    
    def __init__(self):
        super().__init__('collect_edges')
        
        # 파라미터 선언
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('tag_frame_prefix', 'tag_')
        self.declare_parameter('out_edges', './data/edges.jsonl')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('odom_topic', '/odom')
        
        # 품질 파라미터
        self.declare_parameter('dm_min', 40.0)
        self.declare_parameter('dm_good', 70.0)
        self.declare_parameter('px_min', 70.0)
        self.declare_parameter('px_good', 140.0)
        self.declare_parameter('max_hamming', 0)
        
        # 이상치 제거 파라미터
        self.declare_parameter('max_edge_dist', 5.0)
        self.declare_parameter('max_edge_angle', 0.785)  # 45 deg
        
        # 고정밀 파라미터
        self.declare_parameter('velocity_threshold', self.CONFIG['velocity_threshold'])
        self.declare_parameter('min_samples', self.CONFIG['min_samples'])
        self.declare_parameter('target_std_mm', self.CONFIG['target_std_mm'])
        
        # 디버그
        self.declare_parameter('debug', True)
        
        # 파라미터 가져오기
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tag_frame_prefix = self.get_parameter('tag_frame_prefix').value
        self.out_edges_path = self.get_parameter('out_edges').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        
        self.quality_params = QualityParams(
            dm_min=self.get_parameter('dm_min').value,
            dm_good=self.get_parameter('dm_good').value,
            px_min=self.get_parameter('px_min').value,
            px_good=self.get_parameter('px_good').value,
            max_hamming=self.get_parameter('max_hamming').value
        )
        
        self.max_edge_dist = self.get_parameter('max_edge_dist').value
        self.max_edge_angle = self.get_parameter('max_edge_angle').value
        self.debug = self.get_parameter('debug').value
        
        # 고정밀 파라미터
        self.velocity_threshold = self.get_parameter('velocity_threshold').value
        self.min_samples = self.get_parameter('min_samples').value
        self.target_std = self.get_parameter('target_std_mm').value / 1000.0  # mm -> m
        self.iqr_factor = self.CONFIG['outlier_iqr_factor']
        
        # TF 헬퍼
        self.tf_helper = TFHelper(self)
        
        # 출력 파일 준비
        self._prepare_output_file()
        
        # 상태
        self.is_stationary = False
        self.current_velocity = 0.0
        self.frame_count = 0
        self.detected_tags = set()
        
        # Edge 샘플 저장: (tag_i, tag_j) -> list of (dx, dy, dtheta, weight)
        self.edge_samples = defaultdict(list)
        
        # QoS 설정 (센서 데이터)
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
        
        self.sub_odom = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        
        # 상태 출력 타이머
        self.create_timer(2.0, self.print_status)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("고정밀 CollectEdges 노드 시작")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Camera frame: {self.camera_frame}")
        self.get_logger().info(f"  Output: {self.out_edges_path}")
        self.get_logger().info(f"  목표 정밀도: {self.get_parameter('target_std_mm').value:.1f} mm")
        self.get_logger().info(f"  최소 샘플: {self.min_samples}개/edge")
        self.get_logger().info(f"  정지 속도: < {self.velocity_threshold*100:.1f} cm/s")
        self.get_logger().info("")
        self.get_logger().info("📍 로봇을 각 구역에서 정지시키며 태그를 스캔하세요!")
        self.get_logger().info("📍 모든 edge가 '✓'되면 Ctrl+C로 저장")
        self.get_logger().info("=" * 60)
    
    def _prepare_output_file(self):
        """출력 파일 디렉토리 생성"""
        out_dir = os.path.dirname(self.out_edges_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            self.get_logger().info(f"디렉토리 생성: {out_dir}")
        
    def odom_callback(self, msg: Odometry):
        """Odom에서 속도 추출"""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_velocity = math.sqrt(vx*vx + vy*vy)
        self.is_stationary = self.current_velocity < self.velocity_threshold
    
    def detection_callback(self, msg: AprilTagDetectionArray):
        """detection 메시지 콜백"""
        stamp = Time.from_msg(msg.header.stamp)
        self.frame_count += 1
        
        if not self.is_stationary:
            return  # 이동 중에는 수집 안 함
        
        # 1) 각 태그에 대해 TF lookup + 품질 계산
        tag_data: Dict[int, Tuple[SE2, float]] = {}  # {tag_id: (pose, weight)}
        
        for det in msg.detections:
            tag_id = det.id
            
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
            
            se2_pose, _ = result
            tag_data[tag_id] = (se2_pose, weight)
            self.detected_tags.add(tag_id)
        
        # 2) 동시에 보이는 태그 쌍에 대해 엣지 샘플 수집
        tag_ids = sorted(tag_data.keys())
        
        for i, id_i in enumerate(tag_ids):
            for id_j in tag_ids[i+1:]:
                pose_i, w_i = tag_data[id_i]
                pose_j, w_j = tag_data[id_j]
                
                # 상대변환 계산: T_i_j = inv(T_cam_i) * T_cam_j
                rel_pose = se2_between(pose_i, pose_j)
                
                # 엣지 가중치
                edge_weight = compute_edge_weight(w_i, w_j)
                
                # 이상치 체크
                if not is_valid_edge(
                    rel_pose.x, rel_pose.y, rel_pose.theta,
                    max_dist=self.max_edge_dist,
                    max_angle=self.max_edge_angle
                ):
                    continue
                
                # 샘플 저장 (메모리)
                edge_key = (id_i, id_j)
                self.edge_samples[edge_key].append(
                    (rel_pose.x, rel_pose.y, rel_pose.theta, edge_weight)
                )
    
    def _compute_edge_stats(self, samples):
        """Edge 샘플의 통계 계산 (outlier 제거 후)"""
        if len(samples) < 3:
            return None
        
        arr = np.array(samples)  # (n, 4): dx, dy, dtheta, weight
        
        # IQR 기반 outlier 제거 (x, y만)
        combined_mask = np.ones(len(arr), dtype=bool)
        for dim in range(2):
            q1 = np.percentile(arr[:, dim], 25)
            q3 = np.percentile(arr[:, dim], 75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            combined_mask &= (arr[:, dim] >= lower) & (arr[:, dim] <= upper)
        
        clean_arr = arr[combined_mask]
        
        if len(clean_arr) < 3:
            clean_arr = arr  # outlier가 너무 많으면 원본 사용
        
        # Median 사용 (noise-robust)
        median_x = np.median(clean_arr[:, 0])
        median_y = np.median(clean_arr[:, 1])
        
        # Circular median for theta
        sin_sum = np.sum(np.sin(clean_arr[:, 2]))
        cos_sum = np.sum(np.cos(clean_arr[:, 2]))
        median_theta = np.arctan2(sin_sum, cos_sum)
        
        # 평균 가중치
        avg_weight = np.mean(clean_arr[:, 3])
        
        # 표준편차
        std_x = np.std(clean_arr[:, 0])
        std_y = np.std(clean_arr[:, 1])
        
        return {
            'median': (median_x, median_y, median_theta),
            'std': (std_x, std_y),
            'weight': avg_weight,
            'n_total': len(samples),
            'n_clean': len(clean_arr),
        }
    
    def _is_edge_sufficient(self, stats):
        """Edge가 충분한 품질인지 판단"""
        if stats is None:
            return False
        if stats['n_clean'] < self.min_samples:
            return False
        if max(stats['std']) > self.target_std:
            return False
        return True
    
    def print_status(self):
        """현재 수집 상태 출력"""
        status = "🚗 이동 중" if not self.is_stationary else "🛑 정지 (수집 중)"
        
        self.get_logger().info("")
        self.get_logger().info(f"[Frame {self.frame_count}] {status}, 속도: {self.current_velocity*100:.1f} cm/s")
        self.get_logger().info(f"검출된 태그: {sorted(self.detected_tags)}")
        
        if not self.edge_samples:
            self.get_logger().info("  아직 edge 데이터 없음")
            return
        
        # Edge 상태 출력
        sufficient_count = 0
        total_edges = len(self.edge_samples)
        
        self.get_logger().info(f"Edge 수집 현황 ({total_edges}개):")
        
        for edge_key in sorted(self.edge_samples.keys()):
            samples = self.edge_samples[edge_key]
            stats = self._compute_edge_stats(samples)
            
            if stats is None:
                status_str = f"  tag_{edge_key[0]} → tag_{edge_key[1]}: n={len(samples)} ⏳"
            else:
                std_mm = max(stats['std']) * 1000
                is_suff = self._is_edge_sufficient(stats)
                mark = "✓" if is_suff else "⏳"
                if is_suff:
                    sufficient_count += 1
                status_str = f"  tag_{edge_key[0]} → tag_{edge_key[1]}: n={stats['n_clean']}, std={std_mm:.2f}mm {mark}"
            
            self.get_logger().info(status_str)
        
        self.get_logger().info(f"충분한 edge: {sufficient_count}/{total_edges}")
        
        if sufficient_count == total_edges and total_edges > 0:
            self.get_logger().info("🎉 모든 edge 충분! Ctrl+C로 저장하세요.")
    
    def save_edges(self):
        """최종 edge 저장 (outlier 제거 + median)"""
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("edges.jsonl 저장 중...")
        
        saved_count = 0
        
        with open(self.out_edges_path, 'w') as f:
            for (id_i, id_j), samples in sorted(self.edge_samples.items()):
                stats = self._compute_edge_stats(samples)
                
                if stats is None:
                    self.get_logger().warn(f"Edge ({id_i}, {id_j}): 데이터 부족, 건너뜀")
                    continue
                
                dx, dy, dtheta = stats['median']
                
                edge = {
                    "t": 0.0,  # 시간은 의미 없음 (median이므로)
                    "i": id_i,
                    "j": id_j,
                    "dx": round(float(dx), 6),
                    "dy": round(float(dy), 6),
                    "dtheta": round(float(dtheta), 6),
                    "w": round(float(stats['weight']), 4),
                    "n": stats['n_clean'],
                    "std_mm": round(float(max(stats['std']) * 1000), 2)
                }
                
                f.write(json.dumps(edge) + '\n')
                saved_count += 1
                
                std_mm = max(stats['std']) * 1000
            self.get_logger().info(
                    f"  tag_{id_i} → tag_{id_j}: "
                    f"({dx:.4f}, {dy:.4f}), n={stats['n_clean']}, std={std_mm:.2f}mm"
            )
    
        self.get_logger().info("")
        self.get_logger().info(f"✅ 저장 완료: {self.out_edges_path}")
        self.get_logger().info(f"   총 {saved_count}개 edge")
        self.get_logger().info("=" * 60)
    
    def destroy_node(self):
        """노드 종료 시 저장"""
        self.save_edges()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CollectEdgesNode()
    
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
