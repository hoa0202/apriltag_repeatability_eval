#!/usr/bin/env python3
"""
TF 기반 gt_tag_map 자동 생성 스크립트 (고정밀 버전)

특징:
  - 정지 상태에서만 수집 (Odom 속도 체크)
  - 표준편차 수렴 기반 충분성 판단
  - Outlier rejection (IQR)
  - Median 사용 (noise-robust)
  - 실시간 품질 피드백

사용법:
  ros2 run apriltag_repeatability_eval generate_gt_tag_map

시나리오:
  1. 노드 실행
  2. 로봇 이동 → 구역 도착 → 3~5초 정지 → 다음 구역 이동
  3. 모든 edge가 "✓ 충분" 표시되면 Ctrl+C
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from apriltag_msgs.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry
import yaml
import os
import numpy as np
from collections import defaultdict
import math


class GenerateGtTagMapNode(Node):
    # === 설정 ===
    CONFIG = {
        'reference_tag': 3,           # 기준 태그 ID
        'camera_frame': 'camera_link',
        'tag_frame_prefix': 'tag_',
        'odom_topic': '/odom',
        
        # 품질 기준
        'velocity_threshold': 0.005,   # 정지 판정 속도 (m/s)
        'min_samples': 30,             # edge당 최소 샘플 수
        'target_std_mm': 1.0,          # 목표 표준편차 (mm)
        'outlier_iqr_factor': 1.5,     # IQR outlier 제거 계수
        
        # 출력
        'output_path': '',             # 빈 문자열이면 기본 경로
    }
    
    def __init__(self):
        super().__init__('generate_gt_tag_map')
        
        # 파라미터 로드 (CONFIG 기본값 사용)
        self.ref_tag = self.CONFIG['reference_tag']
        self.camera_frame = self.CONFIG['camera_frame']
        self.tag_frame_prefix = self.CONFIG['tag_frame_prefix']
        self.velocity_threshold = self.CONFIG['velocity_threshold']
        self.min_samples = self.CONFIG['min_samples']
        self.target_std = self.CONFIG['target_std_mm'] / 1000.0  # mm -> m
        self.iqr_factor = self.CONFIG['outlier_iqr_factor']
        
        # 출력 경로
        output_path = self.CONFIG['output_path']
        if not output_path:
            from ament_index_python.packages import get_package_share_directory
            try:
                share_dir = get_package_share_directory('apriltag_repeatability_eval')
                ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(share_dir))))
                pkg_path = os.path.join(ws_root, 'src', 'apriltag_repeatability_eval')
            except:
                pkg_path = os.path.expanduser('~/roboro_apriltag_evaluation_ws/src/apriltag_repeatability_eval')
            output_path = os.path.join(pkg_path, 'config', 'gt_tag_map_from_tf.yaml')
        self.output_path = output_path
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 상태
        self.is_stationary = False
        self.current_velocity = 0.0
        self.detected_tags = set()
        self.frame_count = 0
        
        # Edge 데이터: (tag_i, tag_j) -> list of (dx, dy, dtheta)
        self.edge_samples = defaultdict(list)
        
        # 구독
        self.sub_detections = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detection_callback,
            10
        )
        
        self.sub_odom = self.create_subscription(
            Odometry,
            self.CONFIG['odom_topic'],
            self.odom_callback,
            10
        )
        
        # 상태 출력 타이머
        self.create_timer(2.0, self.print_status)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("고정밀 gt_tag_map 생성기 시작")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Reference tag: {self.ref_tag}")
        self.get_logger().info(f"  목표 정밀도: {self.CONFIG['target_std_mm']:.1f} mm")
        self.get_logger().info(f"  최소 샘플: {self.min_samples}개/edge")
        self.get_logger().info(f"  정지 속도: < {self.velocity_threshold*100:.1f} cm/s")
        self.get_logger().info(f"  출력: {self.output_path}")
        self.get_logger().info("")
        self.get_logger().info("📍 로봇을 각 구역에서 정지시키며 태그를 스캔하세요!")
        self.get_logger().info("📍 모든 edge가 '✓'되면 Ctrl+C로 저장")
        self.get_logger().info("=" * 60)
    
    def odom_callback(self, msg: Odometry):
        """Odom에서 속도 추출"""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_velocity = math.sqrt(vx*vx + vy*vy)
        self.is_stationary = self.current_velocity < self.velocity_threshold
    
    def detection_callback(self, msg: AprilTagDetectionArray):
        self.frame_count += 1
        
        if not self.is_stationary:
            return  # 이동 중에는 수집 안 함
        
        # 현재 프레임에서 보이는 모든 태그의 TF 수집
        current_poses = {}  # tag_id -> (x, y, yaw)
        
        for det in msg.detections:
            tag_id = det.id
            tag_frame = f"{self.tag_frame_prefix}{tag_id}"
            
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    tag_frame,
                    Time()
                )
                
                t = transform.transform.translation
                r = transform.transform.rotation
                
                # quaternion -> yaw
                siny_cosp = 2 * (r.w * r.z + r.x * r.y)
                cosy_cosp = 1 - 2 * (r.y * r.y + r.z * r.z)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                
                current_poses[tag_id] = (t.x, t.y, yaw)
                self.detected_tags.add(tag_id)
                
            except Exception:
                pass
        
        # 동시에 보이는 태그 쌍의 상대 변환 계산
        tag_ids = sorted(current_poses.keys())
        for i, id_i in enumerate(tag_ids):
            for id_j in tag_ids[i+1:]:
                # id_i -> id_j 상대 변환
                x_i, y_i, yaw_i = current_poses[id_i]
                x_j, y_j, yaw_j = current_poses[id_j]
                
                # SE2 between: tag_i 좌표계에서 본 tag_j 위치
                cos_i, sin_i = math.cos(-yaw_i), math.sin(-yaw_i)
                dx = (x_j - x_i) * cos_i - (y_j - y_i) * sin_i
                dy = (x_j - x_i) * sin_i + (y_j - y_i) * cos_i
                dtheta = self._wrap_angle(yaw_j - yaw_i)
                
                edge_key = (id_i, id_j)
                self.edge_samples[edge_key].append((dx, dy, dtheta))
    
    def _wrap_angle(self, angle):
        """각도를 -pi ~ pi 범위로"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _compute_edge_stats(self, samples):
        """Edge 샘플의 통계 계산 (outlier 제거 후)"""
        if len(samples) < 3:
            return None
        
        arr = np.array(samples)
        
        # IQR 기반 outlier 제거
        clean_samples = []
        for dim in range(2):  # x, y만 (theta는 별도)
            q1 = np.percentile(arr[:, dim], 25)
            q3 = np.percentile(arr[:, dim], 75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            mask = (arr[:, dim] >= lower) & (arr[:, dim] <= upper)
            if dim == 0:
                combined_mask = mask
            else:
                combined_mask &= mask
        
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
        
        # 표준편차
        std_x = np.std(clean_arr[:, 0])
        std_y = np.std(clean_arr[:, 1])
        
        return {
            'median': (median_x, median_y, median_theta),
            'std': (std_x, std_y),
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
    
    def generate_tag_map(self):
        """최종 tag_map 생성"""
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("gt_tag_map 생성 중...")
        
        # Edge들을 이용해 pose graph 구축
        # BFS로 reference tag부터 시작하여 모든 태그의 절대 좌표 계산
        
        # 먼저 그래프 구성
        graph = defaultdict(dict)  # tag_id -> {neighbor_id: (dx, dy, dtheta)}
        
        for (id_i, id_j), samples in self.edge_samples.items():
            stats = self._compute_edge_stats(samples)
            if stats is None:
                self.get_logger().warn(f"Edge ({id_i}, {id_j}): 데이터 부족, 건너뜀")
                continue
            
            dx, dy, dtheta = stats['median']
            graph[id_i][id_j] = (dx, dy, dtheta)
            # 역방향
            cos_d, sin_d = math.cos(-dtheta), math.sin(-dtheta)
            inv_dx = -dx * cos_d + dy * sin_d
            inv_dy = -dx * sin_d - dy * cos_d
            graph[id_j][id_i] = (inv_dx, inv_dy, -dtheta)
        
        if self.ref_tag not in graph and self.ref_tag not in self.detected_tags:
            self.get_logger().error(f"Reference tag {self.ref_tag} not found!")
            self.get_logger().info(f"검출된 태그: {sorted(self.detected_tags)}")
            return
        
        # BFS로 절대 좌표 계산
        absolute_poses = {self.ref_tag: (0.0, 0.0, 0.0)}
        queue = [self.ref_tag]
        visited = {self.ref_tag}
        
        while queue:
            current = queue.pop(0)
            cx, cy, ctheta = absolute_poses[current]
            
            for neighbor, (dx, dy, dtheta) in graph[current].items():
                if neighbor in visited:
                    continue
                
                # current 좌표계에서 neighbor로의 변환을 global로 변환
                cos_c, sin_c = math.cos(ctheta), math.sin(ctheta)
                nx = cx + dx * cos_c - dy * sin_c
                ny = cy + dx * sin_c + dy * cos_c
                ntheta = self._wrap_angle(ctheta + dtheta)
                
                absolute_poses[neighbor] = (nx, ny, ntheta)
                visited.add(neighbor)
                queue.append(neighbor)
        
        # 결과 출력
        self.get_logger().info(f"계산된 태그 포즈 ({len(absolute_poses)}개):")
        for tag_id in sorted(absolute_poses.keys()):
            x, y, theta = absolute_poses[tag_id]
            edge_key = None
            for ek in self.edge_samples.keys():
                if tag_id in ek:
                    edge_key = ek
                    break
            
            if edge_key:
                stats = self._compute_edge_stats(self.edge_samples[edge_key])
                std_mm = max(stats['std']) * 1000 if stats else 0
                self.get_logger().info(f"  tag_{tag_id}: ({x:.4f}, {y:.4f}, {theta:.4f}), std={std_mm:.2f}mm")
            else:
                self.get_logger().info(f"  tag_{tag_id}: ({x:.4f}, {y:.4f}, {theta:.4f})")
        
        # YAML 저장
        tag_map = {'reference_tag': self.ref_tag, 'tags': {}}
        for tag_id, (x, y, theta) in absolute_poses.items():
            tag_map['tags'][tag_id] = [round(x, 6), round(y, 6), round(theta, 6)]
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            f.write("# Ground Truth Tag Map - Generated from TF (High Precision)\n")
            f.write(f"# Reference tag: {self.ref_tag}\n")
            f.write(f"# Target std: {self.CONFIG['target_std_mm']:.1f} mm\n")
            f.write(f"# Tags: {sorted(absolute_poses.keys())}\n\n")
            yaml.dump(tag_map, f, default_flow_style=False, sort_keys=True)
        
        self.get_logger().info("")
        self.get_logger().info(f"✅ 저장 완료: {self.output_path}")
        self.get_logger().info(f"   총 {len(absolute_poses)}개 태그")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = GenerateGtTagMapNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C 감지, tag_map 생성 중...")
        node.generate_tag_map()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
