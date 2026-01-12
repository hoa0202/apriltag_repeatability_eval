#!/usr/bin/env python3
"""
TF 기반 gt_tag_map 자동 생성 스크립트

로봇을 정지시킨 상태에서 실행하면 실제 TF 데이터를 기반으로
정확한 gt_tag_map.yaml을 생성합니다.

사용법:
  ros2 run apriltag_repeatability_eval generate_gt_tag_map
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from apriltag_msgs.msg import AprilTagDetectionArray
import yaml
import os
import numpy as np
from collections import defaultdict

class GenerateGtTagMapNode(Node):
    def __init__(self):
        super().__init__('generate_gt_tag_map')
        
        # 파라미터
        self.declare_parameter('reference_tag', 3)
        self.declare_parameter('num_samples', 100)
        self.declare_parameter('output_path', '')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('tag_frame_prefix', 'tag_')
        
        self.ref_tag = self.get_parameter('reference_tag').value
        self.num_samples = self.get_parameter('num_samples').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tag_frame_prefix = self.get_parameter('tag_frame_prefix').value
        
        output_path = self.get_parameter('output_path').value
        if not output_path:
            # 기본 경로
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
        
        # 데이터 수집 - 태그 쌍별 상대 좌표
        self.tag_edges = defaultdict(list)  # (tag_i, tag_j) -> list of (rel_x, rel_y, rel_yaw)
        self.sample_count = 0
        self.detected_tags = set()
        
        # 구독
        self.sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detection_callback,
            10
        )
        
        self.get_logger().info(f"gt_tag_map 생성 시작")
        self.get_logger().info(f"  Reference tag: {self.ref_tag}")
        self.get_logger().info(f"  Samples to collect: {self.num_samples}")
        self.get_logger().info(f"  Output: {self.output_path}")
        self.get_logger().info(f"로봇을 이동시키며 모든 태그를 스캔하세요!")
    
    def _get_tag_tf(self, tag_id):
        """태그 TF를 (x, y, yaw)로 반환"""
        tag_frame = f"{self.tag_frame_prefix}{tag_id}"
        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                tag_frame,
                Time()
            )
            t = transform.transform.translation
            r = transform.transform.rotation
            siny_cosp = 2 * (r.w * r.z + r.x * r.y)
            cosy_cosp = 1 - 2 * (r.y * r.y + r.z * r.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            return (t.x, t.y, yaw)
        except:
            return None
    
    def detection_callback(self, msg: AprilTagDetectionArray):
        if self.sample_count >= self.num_samples:
            return
        
        # 현재 프레임에서 보이는 모든 태그의 TF 수집
        frame_tfs = {}
        for det in msg.detections:
            tag_id = det.id
            tag_tf = self._get_tag_tf(tag_id)
            if tag_tf is not None:
                frame_tfs[tag_id] = tag_tf
                self.detected_tags.add(tag_id)
        
        if len(frame_tfs) < 2:
            return  # 최소 2개 태그 필요
        
        # 모든 태그 쌍에 대해 상대 좌표 계산
        tag_ids = list(frame_tfs.keys())
        for i, tag_i in enumerate(tag_ids):
            for tag_j in tag_ids[i+1:]:
                # tag_i -> tag_j 상대 좌표 (SE2 변환)
                xi, yi, yawi = frame_tfs[tag_i]
                xj, yj, yawj = frame_tfs[tag_j]
                
                cos_i = np.cos(-yawi)
                sin_i = np.sin(-yawi)
                dx = xj - xi
                dy = yj - yi
                rel_x = cos_i * dx - sin_i * dy
                rel_y = sin_i * dx + cos_i * dy
                rel_yaw = yawj - yawi
                
                # -pi ~ pi 범위로
                while rel_yaw > np.pi:
                    rel_yaw -= 2 * np.pi
                while rel_yaw < -np.pi:
                    rel_yaw += 2 * np.pi
                
                # 양방향 저장 (tag_i < tag_j로 정렬)
                if tag_i < tag_j:
                    self.tag_edges[(tag_i, tag_j)].append((rel_x, rel_y, rel_yaw))
                else:
                    # 역방향
                    cos_j = np.cos(-yawj)
                    sin_j = np.sin(-yawj)
                    dx_inv = xi - xj
                    dy_inv = yi - yj
                    self.tag_edges[(tag_j, tag_i)].append((
                        cos_j * dx_inv - sin_j * dy_inv,
                        sin_j * dx_inv + cos_j * dy_inv,
                        -rel_yaw
                    ))
        
        self.sample_count += 1
        
        if self.sample_count % 10 == 0:
            self.get_logger().info(f"샘플: {self.sample_count}/{self.num_samples}, 태그: {sorted(self.detected_tags)}, edges: {len(self.tag_edges)}")
        
        if self.sample_count >= self.num_samples:
            self.generate_tag_map()
    
    def generate_tag_map(self):
        self.get_logger().info("gt_tag_map 생성 중 (pose graph 최적화)...")
        
        # Edge 평균 계산
        avg_edges = {}
        for (tag_i, tag_j), measurements in self.tag_edges.items():
            if len(measurements) < 3:
                continue
            arr = np.array(measurements)
            avg_x = np.mean(arr[:, 0])
            avg_y = np.mean(arr[:, 1])
            sin_sum = np.sum(np.sin(arr[:, 2]))
            cos_sum = np.sum(np.cos(arr[:, 2]))
            avg_yaw = np.arctan2(sin_sum, cos_sum)
            std_x = np.std(arr[:, 0])
            std_y = np.std(arr[:, 1])
            
            avg_edges[(tag_i, tag_j)] = (avg_x, avg_y, avg_yaw, len(measurements))
            self.get_logger().info(f"edge ({tag_i}->{tag_j}): ({avg_x:.4f}, {avg_y:.4f}), std=({std_x*100:.1f}, {std_y*100:.1f})cm, n={len(measurements)}")
        
        if len(avg_edges) == 0:
            self.get_logger().error("No edges collected!")
            return
        
        # BFS로 reference tag부터 연결된 태그들의 좌표 계산
        tag_coords = {self.ref_tag: (0.0, 0.0, 0.0)}  # tag_id -> (x, y, yaw)
        visited = {self.ref_tag}
        queue = [self.ref_tag]
        
        while queue:
            current = queue.pop(0)
            cx, cy, cyaw = tag_coords[current]
            cos_c = np.cos(cyaw)
            sin_c = np.sin(cyaw)
            
            # 현재 태그와 연결된 edge 찾기
            for (tag_i, tag_j), (rel_x, rel_y, rel_yaw, _) in avg_edges.items():
                if tag_i == current and tag_j not in visited:
                    # current -> tag_j
                    new_x = cx + cos_c * rel_x - sin_c * rel_y
                    new_y = cy + sin_c * rel_x + cos_c * rel_y
                    new_yaw = cyaw + rel_yaw
                    tag_coords[tag_j] = (new_x, new_y, new_yaw)
                    visited.add(tag_j)
                    queue.append(tag_j)
                elif tag_j == current and tag_i not in visited:
                    # tag_i -> current (역방향)
                    # current -> tag_i = inverse of (tag_i -> current)
                    cos_rel = np.cos(-rel_yaw)
                    sin_rel = np.sin(-rel_yaw)
                    inv_x = -(cos_rel * rel_x - sin_rel * rel_y)
                    inv_y = -(sin_rel * rel_x + cos_rel * rel_y)
                    inv_yaw = -rel_yaw
                    
                    new_x = cx + cos_c * inv_x - sin_c * inv_y
                    new_y = cy + sin_c * inv_x + cos_c * inv_y
                    new_yaw = cyaw + inv_yaw
                    tag_coords[tag_i] = (new_x, new_y, new_yaw)
                    visited.add(tag_i)
                    queue.append(tag_i)
        
        # tag_map 생성
        tag_map = {'reference_tag': self.ref_tag, 'tags': {}}
        for tag_id, (x, y, yaw) in tag_coords.items():
            # -pi ~ pi
            while yaw > np.pi:
                yaw -= 2 * np.pi
            while yaw < -np.pi:
                yaw += 2 * np.pi
            tag_map['tags'][tag_id] = [round(float(x), 6), round(float(y), 6), round(float(yaw), 6)]
            self.get_logger().info(f"tag_{tag_id}: ({x:.4f}, {y:.4f}, {np.degrees(yaw):.1f}°)")
        
        # YAML 저장
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            f.write("# Ground Truth Tag Map - Generated from TF\n")
            f.write(f"# Reference tag: {self.ref_tag}\n")
            f.write(f"# Samples: {self.num_samples}\n")
            f.write(f"# Detected tags: {sorted(tag_map['tags'].keys())}\n\n")
            yaml.dump(tag_map, f, default_flow_style=False, sort_keys=True)
        
        self.get_logger().info(f"저장 완료: {self.output_path}")
        self.get_logger().info(f"총 {len(tag_map['tags'])}개 태그")
        
        # 종료
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = GenerateGtTagMapNode()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
