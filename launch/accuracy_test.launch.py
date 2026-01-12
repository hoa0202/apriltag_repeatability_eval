#!/usr/bin/env python3
"""
정확도 테스트 Launch 파일: AprilTag vs Ground Truth 비교

사용법:
  ros2 launch apriltag_repeatability_eval accuracy_test.launch.py
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
        ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(share_dir))))
        pkg_path = os.path.join(ws_root, 'src', pkg_name)
    except Exception:
        pkg_path = os.path.expanduser('~/roboro_apriltag_evaluation_ws/src/apriltag_repeatability_eval')
    
    default_tag_map = os.path.join(pkg_path, 'config', 'gt_tag_map_from_tf.yaml')
    default_out_csv = os.path.join(pkg_path, 'data', 'accuracy_test.csv')
    
    return LaunchDescription([
        # 카메라 프레임
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_link',
            description='카메라 frame'
        ),
        
        # 태그 맵 경로
        DeclareLaunchArgument(
            'tag_map',
            default_value=default_tag_map,
            description='tag_map.yaml 경로'
        ),
        
        # 출력 CSV
        DeclareLaunchArgument(
            'out_csv',
            default_value=default_out_csv,
            description='출력 CSV 경로'
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
            description='Ground Truth (Odom) 토픽'
        ),
        
        # accuracy_test 노드
        Node(
            package='apriltag_repeatability_eval',
            executable='accuracy_test',
            name='accuracy_test',
            parameters=[{
                'camera_frame': LaunchConfiguration('camera_frame'),
                'tag_frame_prefix': 'tag_',
                'detections_topic': LaunchConfiguration('detections_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'tag_map': LaunchConfiguration('tag_map'),
                'out_csv': LaunchConfiguration('out_csv'),
                'dm_min': 40.0,
                'dm_good': 70.0,
                'px_min': 70.0,
                'px_good': 140.0,
                'max_hamming': 0,
                'debug': True,
            }],
            output='screen'
        ),
    ])
