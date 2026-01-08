#!/usr/bin/env python3
"""
Phase A: 엣지 수집 Launch 파일

사용법:
  ros2 launch apriltag_repeatability_eval phase_a.launch.py
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = os.path.expanduser('~/roboro_april_evaluation_ws/src/apriltag_repeatability_eval')
    default_out = os.path.join(pkg_path, 'data', 'edges.jsonl')
    
    return LaunchDescription([
        # 카메라 프레임
        DeclareLaunchArgument(
            'camera_frame',
            default_value='zed_left_camera_optical_frame',
            description='카메라 optical frame'
        ),
        
        # 출력 파일
        DeclareLaunchArgument(
            'out_edges',
            default_value=default_out,
            description='edges.jsonl 출력 경로'
        ),
        
        # detections 토픽
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/detections',
            description='AprilTag detections 토픽'
        ),
        
        # collect_edges 노드
        Node(
            package='apriltag_repeatability_eval',
            executable='collect_edges',
            name='collect_edges',
            parameters=[{
                'camera_frame': LaunchConfiguration('camera_frame'),
                'tag_frame_prefix': 'tag_',
                'detections_topic': LaunchConfiguration('detections_topic'),
                'out_edges': LaunchConfiguration('out_edges'),
                'dm_min': 40.0,
                'dm_good': 70.0,
                'px_min': 70.0,
                'px_good': 140.0,
                'max_hamming': 0,
                'max_edge_dist': 5.0,
                'max_edge_angle': 0.785,
                'debug': True,
            }],
            output='screen'
        ),
    ])
