#!/usr/bin/env python3
"""
Phase A: 태그-태그 엣지 수집 노드

동시에 보이는 태그 쌍들의 상대변환을 수집하여 edges.jsonl로 저장
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from apriltag_msgs.msg import AprilTagDetectionArray
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ..utils.se2 import SE2, se2_between, wrap_angle
from ..utils.quality import compute_tag_weight, compute_edge_weight, is_valid_edge, QualityParams
from ..utils.tf_utils import TFHelper, transform_to_se2


class CollectEdgesNode(Node):
    """Phase A: 태그-태그 엣지 수집 노드"""
    
    def __init__(self):
        super().__init__('collect_edges')
        
        # 파라미터 선언
        self.declare_parameter('camera_frame', 'zed_left_camera_optical_frame')
        self.declare_parameter('tag_frame_prefix', 'tag_')
        self.declare_parameter('out_edges', './data/edges.jsonl')
        self.declare_parameter('detections_topic', '/detections')
        
        # 품질 파라미터
        self.declare_parameter('dm_min', 40.0)
        self.declare_parameter('dm_good', 70.0)
        self.declare_parameter('px_min', 70.0)
        self.declare_parameter('px_good', 140.0)
        self.declare_parameter('max_hamming', 0)
        
        # 이상치 제거 파라미터
        self.declare_parameter('max_edge_dist', 5.0)
        self.declare_parameter('max_edge_angle', 0.785)  # 45 deg
        
        # 디버그
        self.declare_parameter('debug', True)
        
        # 파라미터 가져오기
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tag_frame_prefix = self.get_parameter('tag_frame_prefix').value
        self.out_edges_path = self.get_parameter('out_edges').value
        self.detections_topic = self.get_parameter('detections_topic').value
        
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
        
        # TF 헬퍼
        self.tf_helper = TFHelper(self)
        
        # 출력 파일 준비
        self._prepare_output_file()
        
        # 통계
        self.frame_count = 0
        self.edge_count = 0
        self.tags_per_frame: List[int] = []
        
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
        
        self.get_logger().info(f"CollectEdges 노드 시작")
        self.get_logger().info(f"  Camera frame: {self.camera_frame}")
        self.get_logger().info(f"  Tag frame prefix: {self.tag_frame_prefix}")
        self.get_logger().info(f"  Output: {self.out_edges_path}")
        self.get_logger().info(f"  Detections topic: {self.detections_topic}")
    
    def _prepare_output_file(self):
        """출력 파일 디렉토리 생성"""
        out_dir = os.path.dirname(self.out_edges_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            self.get_logger().info(f"디렉토리 생성: {out_dir}")
        
        # 파일 초기화 (기존 내용 유지하지 않음 - 새로 시작)
        # 주석: 기존 파일에 append 하려면 아래 줄 주석 처리
        # with open(self.out_edges_path, 'w') as f:
        #     pass
    
    def detection_callback(self, msg: AprilTagDetectionArray):
        """detection 메시지 콜백"""
        stamp = Time.from_msg(msg.header.stamp)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.frame_count += 1
        
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
        
        # 통계 업데이트
        self.tags_per_frame.append(len(tag_data))
        
        # 2) 동시에 보이는 태그 쌍에 대해 엣지 생성
        tag_ids = sorted(tag_data.keys())
        edges_this_frame = 0
        
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
                
                # 엣지 저장
                edge = {
                    "t": timestamp,
                    "i": id_i,
                    "j": id_j,
                    "dx": round(rel_pose.x, 6),
                    "dy": round(rel_pose.y, 6),
                    "dtheta": round(rel_pose.theta, 6),
                    "w": round(edge_weight, 4)
                }
                
                self._write_edge(edge)
                edges_this_frame += 1
                self.edge_count += 1
        
        # 디버그 출력
        if self.debug and self.frame_count % 30 == 0:
            avg_tags = sum(self.tags_per_frame[-100:]) / min(len(self.tags_per_frame), 100)
            fail_rate = self.tf_helper.get_fail_rate() * 100
            self.get_logger().info(
                f"[Frame {self.frame_count}] "
                f"Tags: {len(tag_data)}, Edges: {edges_this_frame}, "
                f"Total edges: {self.edge_count}, "
                f"Avg tags/frame: {avg_tags:.1f}, "
                f"TF fail: {fail_rate:.1f}%"
            )
    
    def _write_edge(self, edge: dict):
        """엣지를 파일에 append"""
        with open(self.out_edges_path, 'a') as f:
            f.write(json.dumps(edge) + '\n')
    
    def destroy_node(self):
        """노드 종료 시 통계 출력"""
        self.get_logger().info("="*50)
        self.get_logger().info("CollectEdges 종료 통계:")
        self.get_logger().info(f"  총 프레임: {self.frame_count}")
        self.get_logger().info(f"  총 엣지: {self.edge_count}")
        if self.tags_per_frame:
            self.get_logger().info(f"  평균 태그/프레임: {sum(self.tags_per_frame)/len(self.tags_per_frame):.2f}")
        self.get_logger().info(f"  TF 실패율: {self.tf_helper.get_fail_rate()*100:.1f}%")
        self.get_logger().info(f"  출력 파일: {self.out_edges_path}")
        self.get_logger().info("="*50)
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
