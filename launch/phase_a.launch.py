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
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # share 디렉토리에서 workspace root 역산 -> src 폴더로
    pkg_name = 'apriltag_repeatability_eval'
    try:
        share_dir = get_package_share_directory(pkg_name)
        # share_dir: .../install/pkg_name/share/pkg_name
        # -> .../install/pkg_name/share -> .../install/pkg_name -> .../install -> ws_root
        ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(share_dir))))
        pkg_path = os.path.join(ws_root, 'src', pkg_name)
    except Exception:
        pkg_path = os.path.expanduser('~/roboro_apriltag_evaluation_ws/src/apriltag_repeatability_eval')
    default_out = os.path.join(pkg_path, 'data', 'edges.jsonl')
    
    return LaunchDescription([
        # 카메라 프레임
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_link',
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
        
        # odom 토픽
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Odometry 토픽 (정지 감지용)'
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
                'odom_topic': LaunchConfiguration('odom_topic'),
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
