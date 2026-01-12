#!/usr/bin/env python3
"""
Phase B: 궤적 기록 Launch 파일

사용법:
  # 자동 번호 (run_01, run_02, ... 순차적)
  ros2 launch apriltag_repeatability_eval phase_b.launch.py

  # 수동 지정
  ros2 launch apriltag_repeatability_eval phase_b.launch.py out_csv:=/path/to/run_05.csv
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
    default_tag_map = os.path.join(pkg_path, 'config', 'tag_map.yaml')
    default_data_dir = os.path.join(pkg_path, 'data')
    
    return LaunchDescription([
        # 카메라 프레임
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_link',
            description='카메라 optical frame'
        ),
        
        # 태그 맵 경로
        DeclareLaunchArgument(
            'tag_map',
            default_value=default_tag_map,
            description='tag_map.yaml 경로'
        ),
        
        # 데이터 디렉토리
        DeclareLaunchArgument(
            'data_dir',
            default_value=default_data_dir,
            description='데이터 저장 디렉토리'
        ),
        
        # 출력 CSV ('auto' = 자동 번호)
        DeclareLaunchArgument(
            'out_csv',
            default_value='auto',
            description='출력 CSV 경로 (auto = 자동 번호)'
        ),
        
        # detections 토픽
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/detections',
            description='AprilTag detections 토픽'
        ),
        
        # localize_and_record 노드
        Node(
            package='apriltag_repeatability_eval',
            executable='localize_and_record',
            name='localize_and_record',
            parameters=[{
                'camera_frame': LaunchConfiguration('camera_frame'),
                'tag_frame_prefix': 'tag_',
                'detections_topic': LaunchConfiguration('detections_topic'),
                'tag_map': LaunchConfiguration('tag_map'),
                'out_csv': LaunchConfiguration('out_csv'),
                'data_dir': LaunchConfiguration('data_dir'),
                'dm_min': 40.0,
                'dm_good': 70.0,
                'px_min': 70.0,
                'px_good': 140.0,
                'max_hamming': 0,
                'publish_path': True,
                'path_frame': 'tag_0',
                'debug': True,
            }],
            output='screen'
        ),
    ])
