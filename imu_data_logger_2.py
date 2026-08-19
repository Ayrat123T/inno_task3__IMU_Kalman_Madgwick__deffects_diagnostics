#!/usr/bin/env python3
"""IMU Data Logger and Comprehensive Analysis Node.

Subscribes: /ouster/imu (reference)
            /ouster/imu_defective (defective)

Performs comprehensive analysis including:
- Frequency and timestamp analysis
- Signal statistics (mean, std, RMS, min, max, skewness, kurtosis)
- Static bias calculation (stationary periods)
- Outlier detection
- Frame_id validation
- Covariance matrix analysis
- Correlation analysis
- Spectral analysis (FFT)
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, periodogram

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


class IMUDataLogger(Node):
    def __init__(self) -> None:
        super().__init__("imu_data_logger")

        # Parameters
        self.declare_parameter("record_duration", 60.0)
        self.declare_parameter("reference_topic", "/ouster/imu")
        self.declare_parameter("defective_topic", "/ouster/imu_defective")
        self.declare_parameter("output_dir", str(os.path.expanduser("/home/user/ros2_ws/src/inno/task3/imu_analysis_2")))
        self.declare_parameter("wait_for_topics", 10.0)
        self.declare_parameter("static_threshold_accel", 0.05)  # m/s²
        self.declare_parameter("static_threshold_gyro", 0.01)   # rad/s
        self.declare_parameter("outlier_std_threshold", 5.0)    # number of std deviations

        self.record_duration = float(self.get_parameter("record_duration").value)
        self.reference_topic = str(self.get_parameter("reference_topic").value)
        self.defective_topic = str(self.get_parameter("defective_topic").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.wait_for_topics = float(self.get_parameter("wait_for_topics").value)
        self.static_threshold_accel = float(self.get_parameter("static_threshold_accel").value)
        self.static_threshold_gyro = float(self.get_parameter("static_threshold_gyro").value)
        self.outlier_threshold = float(self.get_parameter("outlier_std_threshold").value)

        # Data storage
        self.reference_data = {
            'timestamps': [],
            'timestamp_raw': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
            'orientation_w': [], 'orientation_x': [], 'orientation_y': [], 'orientation_z': [],
            'accel_cov': [], 'gyro_cov': [], 'orient_cov': [],
            'frame_id': [],
            'header_stamp_sec': [], 'header_stamp_nsec': []
        }
        
        self.defective_data = {
            'timestamps': [],
            'timestamp_raw': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
            'orientation_w': [], 'orientation_x': [], 'orientation_y': [], 'orientation_z': [],
            'accel_cov': [], 'gyro_cov': [], 'orient_cov': [],
            'frame_id': [],
            'header_stamp_sec': [], 'header_stamp_nsec': []
        }

        # Statistics storage
        self.reference_stats = {}
        self.defective_stats = {}
        self.comparison_stats = {}
        self.frequency_stats = {}
        self.outlier_stats = {}
        self.covariance_stats = {}
        self.static_bias = {}

        # Subscribers
        self.ref_sub = self.create_subscription(
            Imu, self.reference_topic, self._reference_callback, 10
        )
        self.def_sub = self.create_subscription(
            Imu, self.defective_topic, self._defective_callback, 10
        )

        # State
        self.start_time = None
        self.recording_active = False
        self.recording_started = False
        self.ref_msg_count = 0
        self.def_msg_count = 0
        self.ref_prev_timestamp = None
        self.def_prev_timestamp = None
        self.ref_dt_list = []
        self.def_dt_list = []
        
        # Check topics
        self.get_logger().info("=" * 70)
        self.get_logger().info("IMU Data Logger with Comprehensive Analysis")
        self.get_logger().info(f"Recording duration: {self.record_duration} seconds")
        self.get_logger().info(f"Reference topic: {self.reference_topic}")
        self.get_logger().info(f"Defective topic: {self.defective_topic}")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("=" * 70)
        
        self.check_timer = self.create_timer(0.5, self._check_topics)
        self.get_logger().info("Waiting for topics to become available...")

    def _check_topics(self) -> None:
        """Check if topics are available and start recording."""
        if self.recording_started:
            return
            
        if not hasattr(self, '_check_start_time'):
            self._check_start_time = time.time()
        
        elapsed = time.time() - self._check_start_time
        if elapsed > self.wait_for_topics:
            self.get_logger().warn(f"Timeout waiting for topics after {self.wait_for_topics} seconds. Exiting...")
            rclpy.shutdown()
            sys.exit(0)
        
        if self.ref_msg_count > 0 and self.def_msg_count > 0:
            self.get_logger().info(f"Both topics are active! Reference: {self.ref_msg_count} msgs, Defective: {self.def_msg_count} msgs")
            self.get_logger().info("Starting recording in 5 seconds...")
            self.check_timer.cancel()
            
            self.start_timer = self.create_timer(5.0, self._start_recording)
            self.recording_started = True

    def _start_recording(self) -> None:
        """Start recording data."""
        if self.recording_active:
            return
        
        self.start_time = time.time()
        self.recording_active = True
        self.start_timer.cancel()
        self.get_logger().info(f"Recording started! Will run for {self.record_duration} seconds.")
        
        self.stop_timer = self.create_timer(self.record_duration, self._stop_recording)

    def _stop_recording(self) -> None:
        """Stop recording and analyze data."""
        if not self.recording_active:
            return
        
        self.recording_active = False
        self.stop_timer.cancel()
        self.get_logger().info(f"Recording complete! Collected {len(self.reference_data['timestamps'])} reference samples and {len(self.defective_data['timestamps'])} defective samples.")
        
        if len(self.reference_data['timestamps']) < 10 or len(self.defective_data['timestamps']) < 10:
            self.get_logger().error("Not enough data collected!")
            rclpy.shutdown()
            sys.exit(0)
        
        self._analyze_all()
        self._generate_comprehensive_report()
        self.get_logger().info("Analysis complete. Exiting...")
        rclpy.shutdown()
        sys.exit(0)

    def _reference_callback(self, msg: Imu) -> None:
        """Callback for reference IMU data."""
        self.ref_msg_count += 1
        
        if not self.recording_active:
            return
        
        timestamp = self._get_seconds(msg.header.stamp)
        rel_time = timestamp - self.start_time
        
        # Frequency analysis
        if self.ref_prev_timestamp is not None:
            dt = timestamp - self.ref_prev_timestamp
            if dt > 0:
                self.ref_dt_list.append(dt)
        self.ref_prev_timestamp = timestamp
        
        # Store data
        self.reference_data['timestamps'].append(rel_time)
        self.reference_data['timestamp_raw'].append(timestamp)
        self.reference_data['header_stamp_sec'].append(msg.header.stamp.sec)
        self.reference_data['header_stamp_nsec'].append(msg.header.stamp.nanosec)
        self.reference_data['frame_id'].append(msg.header.frame_id)
        
        self.reference_data['accel_x'].append(msg.linear_acceleration.x)
        self.reference_data['accel_y'].append(msg.linear_acceleration.y)
        self.reference_data['accel_z'].append(msg.linear_acceleration.z)
        
        self.reference_data['gyro_x'].append(msg.angular_velocity.x)
        self.reference_data['gyro_y'].append(msg.angular_velocity.y)
        self.reference_data['gyro_z'].append(msg.angular_velocity.z)
        
        self.reference_data['orientation_w'].append(msg.orientation.w)
        self.reference_data['orientation_x'].append(msg.orientation.x)
        self.reference_data['orientation_y'].append(msg.orientation.y)
        self.reference_data['orientation_z'].append(msg.orientation.z)
        
        # Covariance matrices (full 3x3)
        self.reference_data['accel_cov'].append(np.array(msg.linear_acceleration_covariance).reshape(3, 3))
        self.reference_data['gyro_cov'].append(np.array(msg.angular_velocity_covariance).reshape(3, 3))
        self.reference_data['orient_cov'].append(np.array(msg.orientation_covariance).reshape(3, 3))

    def _defective_callback(self, msg: Imu) -> None:
        """Callback for defective IMU data."""
        self.def_msg_count += 1
        
        if not self.recording_active:
            return
        
        timestamp = self._get_seconds(msg.header.stamp)
        rel_time = timestamp - self.start_time
        
        # Frequency analysis
        if self.def_prev_timestamp is not None:
            dt = timestamp - self.def_prev_timestamp
            if dt > 0:
                self.def_dt_list.append(dt)
        self.def_prev_timestamp = timestamp
        
        # Store data
        self.defective_data['timestamps'].append(rel_time)
        self.defective_data['timestamp_raw'].append(timestamp)
        self.defective_data['header_stamp_sec'].append(msg.header.stamp.sec)
        self.defective_data['header_stamp_nsec'].append(msg.header.stamp.nanosec)
        self.defective_data['frame_id'].append(msg.header.frame_id)
        
        self.defective_data['accel_x'].append(msg.linear_acceleration.x)
        self.defective_data['accel_y'].append(msg.linear_acceleration.y)
        self.defective_data['accel_z'].append(msg.linear_acceleration.z)
        
        self.defective_data['gyro_x'].append(msg.angular_velocity.x)
        self.defective_data['gyro_y'].append(msg.angular_velocity.y)
        self.defective_data['gyro_z'].append(msg.angular_velocity.z)
        
        self.defective_data['orientation_w'].append(msg.orientation.w)
        self.defective_data['orientation_x'].append(msg.orientation.x)
        self.defective_data['orientation_y'].append(msg.orientation.y)
        self.defective_data['orientation_z'].append(msg.orientation.z)
        
        # Covariance matrices
        self.defective_data['accel_cov'].append(np.array(msg.linear_acceleration_covariance).reshape(3, 3))
        self.defective_data['gyro_cov'].append(np.array(msg.angular_velocity_covariance).reshape(3, 3))
        self.defective_data['orient_cov'].append(np.array(msg.orientation_covariance).reshape(3, 3))

    def _get_seconds(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _analyze_all(self) -> None:
        """Perform comprehensive analysis."""
        self.get_logger().info("Starting comprehensive analysis...")
        
        # 1. Basic statistics
        self._analyze_statistics()
        
        # 2. Frequency analysis
        self._analyze_frequency()
        
        # 3. Static bias
        self._analyze_static_bias()
        
        # 4. Outlier detection
        self._analyze_outliers()
        
        # 5. Covariance analysis
        self._analyze_covariance()
        
        # 6. Frame ID analysis
        self._analyze_frame_id()
        
        # 7. Correlation analysis
        self._analyze_correlations()

    def _analyze_statistics(self) -> None:
        """Compute comprehensive statistics for all signals."""
        self.get_logger().info("Analyzing signal statistics...")
        
        for dataset_name, dataset in [('reference', self.reference_data), ('defective', self.defective_data)]:
            stats_dict = {}
            for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                data = np.array(dataset[key])
                if len(data) > 0:
                    stats_dict[key] = {
                        'count': len(data),
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'var': np.var(data),
                        'min': np.min(data),
                        'max': np.max(data),
                        'range': np.max(data) - np.min(data),
                        'rms': np.sqrt(np.mean(data**2)),
                        'skewness': stats.skew(data) if len(data) > 2 else 0,
                        'kurtosis': stats.kurtosis(data) if len(data) > 3 else 0,
                        'median': np.median(data),
                        'q25': np.percentile(data, 25),
                        'q75': np.percentile(data, 75),
                        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
                    }
            
            if dataset_name == 'reference':
                self.reference_stats = stats_dict
            else:
                self.defective_stats = stats_dict

    def _analyze_frequency(self) -> None:
        """Analyze message frequency and timing."""
        self.get_logger().info("Analyzing frequency and timing...")
        
        for name, dt_list in [('reference', self.ref_dt_list), ('defective', self.def_dt_list)]:
            if len(dt_list) > 1:
                dt_array = np.array(dt_list)
                freq = 1.0 / np.mean(dt_array)
                self.frequency_stats[name] = {
                    'mean_dt': np.mean(dt_array),
                    'std_dt': np.std(dt_array),
                    'min_dt': np.min(dt_array),
                    'max_dt': np.max(dt_array),
                    'mean_freq': freq,
                    'min_freq': 1.0 / np.max(dt_array),
                    'max_freq': 1.0 / np.min(dt_array),
                    'jitter': np.std(dt_array) * 1000,  # ms
                    'dropped_count': 0,  # Will be calculated
                    'expected_count': int(self.record_duration * freq),
                    'actual_count': len(dt_list) + 1,
                }
                
                # Detect dropped messages (gaps > 2x expected dt)
                expected_dt = np.mean(dt_array)
                drops = np.sum(dt_array > 2.5 * expected_dt)
                self.frequency_stats[name]['dropped_count'] = drops

    def _analyze_static_bias(self) -> None:
        """Detect stationary periods and compute bias."""
        self.get_logger().info("Analyzing static bias...")
        
        for name, dataset in [('reference', self.reference_data), ('defective', self.defective_data)]:
            accel = np.column_stack([dataset['accel_x'], dataset['accel_y'], dataset['accel_z']])
            gyro = np.column_stack([dataset['gyro_x'], dataset['gyro_y'], dataset['gyro_z']])
            
            # Detect static periods (low accel and gyro variance)
            accel_mag = np.linalg.norm(accel, axis=1)
            gyro_mag = np.linalg.norm(gyro, axis=1)
            
            accel_std = np.std(accel_mag)
            gyro_std = np.std(gyro_mag)
            
            # Find static regions
            static_indices = np.where((accel_mag < np.mean(accel_mag) + 2*accel_std) & 
                                     (gyro_mag < 0.5))[0]
            
            if len(static_indices) > 10:
                static_accel = accel[static_indices]
                static_gyro = gyro[static_indices]
                
                self.static_bias[name] = {
                    'accel_bias': np.mean(static_accel, axis=0).tolist(),
                    'accel_bias_std': np.std(static_accel, axis=0).tolist(),
                    'gyro_bias': np.mean(static_gyro, axis=0).tolist(),
                    'gyro_bias_std': np.std(static_gyro, axis=0).tolist(),
                    'static_samples': len(static_indices),
                    'static_percentage': 100 * len(static_indices) / len(accel)
                }
            else:
                self.static_bias[name] = {
                    'error': 'No static period detected'
                }

    def _analyze_outliers(self) -> None:
        """Detect outliers using multiple methods."""
        self.get_logger().info("Detecting outliers...")
        
        for name, dataset in [('reference', self.reference_data), ('defective', self.defective_data)]:
            outlier_stats = {}
            
            for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                data = np.array(dataset[key])
                mean = np.mean(data)
                std = np.std(data)
                
                # Z-score method
                z_scores = np.abs((data - mean) / std)
                z_outliers = np.where(z_scores > self.outlier_threshold)[0]
                
                # IQR method
                q25 = np.percentile(data, 25)
                q75 = np.percentile(data, 75)
                iqr = q75 - q25
                iqr_outliers = np.where((data < q25 - 1.5*iqr) | (data > q75 + 1.5*iqr))[0]
                
                # Combined outliers
                all_outliers = np.union1d(z_outliers, iqr_outliers)
                
                outlier_stats[key] = {
                    'z_score_outliers': len(z_outliers),
                    'z_score_percent': 100 * len(z_outliers) / len(data),
                    'iqr_outliers': len(iqr_outliers),
                    'iqr_percent': 100 * len(iqr_outliers) / len(data),
                    'total_outliers': len(all_outliers),
                    'total_percent': 100 * len(all_outliers) / len(data),
                    'outlier_indices': all_outliers.tolist()[:10],  # First 10 only
                    'min_valid': mean - 3*std,
                    'max_valid': mean + 3*std,
                }
            
            self.outlier_stats[name] = outlier_stats

    def _analyze_covariance(self) -> None:
        """Analyze covariance matrices."""
        self.get_logger().info("Analyzing covariance matrices...")
        
        for name, dataset in [('reference', self.reference_data), ('defective', self.defective_data)]:
            cov_stats = {}
            
            for cov_type in ['accel_cov', 'gyro_cov', 'orient_cov']:
                cov_matrices = dataset[cov_type]
                if len(cov_matrices) == 0:
                    continue
                
                # Check if covariance is valid (positive definite)
                valid_cov = []
                for cov in cov_matrices:
                    if isinstance(cov, np.ndarray) and cov.shape == (3, 3):
                        # Check if symmetric
                        is_symmetric = np.allclose(cov, cov.T)
                        # Check if positive definite (all eigenvalues > 0)
                        if is_symmetric:
                            try:
                                eigenvals = np.linalg.eigvalsh(cov)
                                is_positive = np.all(eigenvals > 0)
                            except:
                                is_positive = False
                        else:
                            is_positive = False
                        
                        valid_cov.append({
                            'is_valid': is_symmetric and is_positive,
                            'eigenvalues': np.linalg.eigvalsh(cov).tolist() if is_symmetric else [],
                            'condition_number': np.linalg.cond(cov) if is_symmetric else float('inf'),
                            'trace': np.trace(cov),
                            'det': np.linalg.det(cov) if is_symmetric else 0
                        })
                
                if valid_cov:
                    valid_flags = [c['is_valid'] for c in valid_cov]
                    cov_stats[cov_type] = {
                        'total_samples': len(valid_cov),
                        'valid_cov_count': sum(valid_flags),
                        'invalid_cov_count': len(valid_cov) - sum(valid_flags),
                        'invalid_percent': 100 * (len(valid_cov) - sum(valid_flags)) / len(valid_cov),
                        'avg_trace': np.mean([c['trace'] for c in valid_cov if c['is_valid']]),
                        'avg_condition': np.mean([c['condition_number'] for c in valid_cov if c['is_valid']]),
                        'avg_det': np.mean([c['det'] for c in valid_cov if c['is_valid']]),
                        'sample': valid_cov[0] if valid_cov else {}
                    }
            
            self.covariance_stats[name] = cov_stats

    def _analyze_frame_id(self) -> None:
        """Analyze frame_id consistency."""
        self.get_logger().info("Analyzing frame_id...")
        
        for name, dataset in [('reference', self.reference_data), ('defective', self.defective_data)]:
            frame_ids = dataset['frame_id']
            unique_frames = set(frame_ids)
            
            self.comparison_stats[f'{name}_frame_id'] = {
                'unique_frames': list(unique_frames),
                'frame_count': len(unique_frames),
                'most_common': max(set(frame_ids), key=frame_ids.count) if frame_ids else None,
                'most_common_count': frame_ids.count(max(set(frame_ids), key=frame_ids.count)) if frame_ids else 0,
                'consistency': 100 * frame_ids.count(max(set(frame_ids), key=frame_ids.count)) / len(frame_ids) if frame_ids else 0,
                'changes_detected': len([i for i in range(1, len(frame_ids)) if frame_ids[i] != frame_ids[i-1]])
            }

    def _analyze_correlations(self) -> None:
        """Analyze correlations between signals."""
        self.get_logger().info("Analyzing signal correlations...")
        
        for name, dataset in [('reference', self.reference_data), ('defective', self.defective_data)]:
            # Create data matrix
            signals = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            data_matrix = np.column_stack([dataset[s] for s in signals])
            
            if len(data_matrix) > 0:
                corr_matrix = np.corrcoef(data_matrix.T)
                
                self.comparison_stats[f'{name}_correlations'] = {
                    'signals': signals,
                    'correlation_matrix': corr_matrix.tolist(),
                    'max_correlation': np.max(corr_matrix[corr_matrix < 0.999]),
                    'min_correlation': np.min(corr_matrix),
                    'mean_correlation': np.mean(corr_matrix[corr_matrix < 0.999])
                }

    def _generate_comprehensive_report(self) -> None:
        """Generate comprehensive report with plots and statistics."""
        self.get_logger().info("Generating comprehensive report...")
        
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save statistics as JSON
        self._save_json_report(timestamp)
        
        # Generate plots if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            try:
                self._generate_enhanced_plots(timestamp)
            except Exception as e:
                self.get_logger().error(f"Error generating plots: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        self._print_summary()

    def _save_json_report(self, timestamp: str) -> None:
        """Save comprehensive report as JSON."""
        report = {
            'timestamp': timestamp,
            'recording_duration': self.record_duration,
            'reference_topic': self.reference_topic,
            'defective_topic': self.defective_topic,
            'sample_counts': {
                'reference': len(self.reference_data['timestamps']),
                'defective': len(self.defective_data['timestamps'])
            },
            'statistics': {
                'reference': self.reference_stats,
                'defective': self.defective_stats,
                'comparison': self.comparison_stats
            },
            'frequency': self.frequency_stats,
            'static_bias': self.static_bias,
            'outliers': self.outlier_stats,
            'covariance': self.covariance_stats,
            'frame_id': self.comparison_stats.get('reference_frame_id', {}),
            'defective_frame_id': self.comparison_stats.get('defective_frame_id', {})
        }
        
        # Save to file
        json_file = os.path.join(self.output_dir, f'report_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.get_logger().info(f"Saved JSON report to {json_file}")
        
        # Also save as text for readability
        txt_file = os.path.join(self.output_dir, f'report_{timestamp}.txt')
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("IMU COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Recording Duration: {self.record_duration} seconds\n")
            f.write(f"Reference Samples: {len(self.reference_data['timestamps'])}\n")
            f.write(f"Defective Samples: {len(self.defective_data['timestamps'])}\n\n")
            
            # Statistics
            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                if key in self.reference_stats:
                    ref = self.reference_stats[key]
                    def_ = self.defective_stats.get(key, {})
                    f.write(f"\n{key.upper()}:\n")
                    f.write(f"  Reference: Mean={ref['mean']:.4f}, Std={ref['std']:.4f}, RMS={ref['rms']:.4f}\n")
                    if def_:
                        f.write(f"  Defective: Mean={def_['mean']:.4f}, Std={def_['std']:.4f}, RMS={def_['rms']:.4f}\n")
                        f.write(f"  Difference: Mean={def_['mean']-ref['mean']:.4f}, Std={def_['std']-ref['std']:.4f}\n")
            
            # Frequency
            if 'reference' in self.frequency_stats:
                ref_freq = self.frequency_stats['reference']
                f.write("\nFREQUENCY ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Reference: {ref_freq['mean_freq']:.1f} Hz (std: {ref_freq['std_dt']*1000:.1f} ms jitter)\n")
                if 'defective' in self.frequency_stats:
                    def_freq = self.frequency_stats['defective']
                    f.write(f"  Defective: {def_freq['mean_freq']:.1f} Hz (std: {def_freq['std_dt']*1000:.1f} ms jitter)\n")
            
            # Static bias
            f.write("\nSTATIC BIAS\n")
            f.write("-" * 80 + "\n")
            for name in ['reference', 'defective']:
                if name in self.static_bias:
                    bias = self.static_bias[name]
                    if 'error' not in bias:
                        f.write(f"  {name.capitalize()}:\n")
                        f.write(f"    Accel Bias: [{bias['accel_bias'][0]:.4f}, {bias['accel_bias'][1]:.4f}, {bias['accel_bias'][2]:.4f}] m/s²\n")
                        f.write(f"    Gyro Bias:  [{bias['gyro_bias'][0]:.4f}, {bias['gyro_bias'][1]:.4f}, {bias['gyro_bias'][2]:.4f}] rad/s\n")
            
            # Outliers
            f.write("\nOUTLIER ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for name in ['reference', 'defective']:
                if name in self.outlier_stats:
                    stats = self.outlier_stats[name]
                    total_outliers = sum([s['total_outliers'] for s in stats.values()])
                    total_samples = sum([self.reference_stats.get(k, {}).get('count', 0) for k in stats.keys()])
                    f.write(f"  {name.capitalize()}: {total_outliers} outliers ({100*total_outliers/total_samples:.1f}%)\n")
            
            # Covariance
            f.write("\nCOVARIANCE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for name in ['reference', 'defective']:
                if name in self.covariance_stats:
                    cov_stats = self.covariance_stats[name]
                    for cov_type, stats in cov_stats.items():
                        if 'valid_cov_count' in stats:
                            f.write(f"  {name.capitalize()} - {cov_type}:\n")
                            f.write(f"    Valid: {stats['valid_cov_count']}/{stats['total_samples']} ({100-stats['invalid_percent']:.1f}%)\n")
                            if 'avg_trace' in stats:
                                f.write(f"    Avg Trace: {stats['avg_trace']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        self.get_logger().info(f"Saved text report to {txt_file}")

    def _generate_enhanced_plots(self, timestamp: str) -> None:
        """Generate enhanced plots with all analysis results."""
        self.get_logger().info("Generating plots...")
        
        # Configure matplotlib
        rcParams['font.size'] = 10
        rcParams['figure.figsize'] = (14, 10)
        
        # Plot 1: Time series comparison with statistical annotations
        fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
        
        for i, (axis, label) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
            # Acceleration
            ax = axes1[0, i]
            ref_data = np.array(self.reference_data[f'accel_{axis}'])
            def_data = np.array(self.defective_data[f'accel_{axis}'])
            min_len = min(len(ref_data), len(def_data))
            timestamps = self.reference_data['timestamps'][:min_len]
            
            ax.plot(timestamps, ref_data[:min_len], 'g-', label='Reference', alpha=0.7, linewidth=1.5)
            ax.plot(timestamps, def_data[:min_len], 'r-', label='Defective', alpha=0.7, linewidth=1.5)
            
            # Add static bias annotation
            if 'reference' in self.static_bias and 'accel_bias' in self.static_bias['reference']:
                bias = self.static_bias['reference']['accel_bias'][i]
                ax.axhline(y=bias, color='g', linestyle='--', alpha=0.5, label=f'Ref Bias: {bias:.3f}')
            if 'defective' in self.static_bias and 'accel_bias' in self.static_bias['defective']:
                bias_def = self.static_bias['defective']['accel_bias'][i]
                ax.axhline(y=bias_def, color='r', linestyle='--', alpha=0.5, label=f'Def Bias: {bias_def:.3f}')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.set_title(f'Acceleration {label}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            # Gyroscope
            ax = axes1[1, i]
            ref_data = np.array(self.reference_data[f'gyro_{axis}'])
            def_data = np.array(self.defective_data[f'gyro_{axis}'])
            min_len = min(len(ref_data), len(def_data))
            
            ax.plot(timestamps, ref_data[:min_len], 'b-', label='Reference', alpha=0.7, linewidth=1.5)
            ax.plot(timestamps, def_data[:min_len], 'r-', label='Defective', alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angular Velocity (rad/s)')
            ax.set_title(f'Gyroscope {label}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        fig1.savefig(os.path.join(self.output_dir, f'01_timeseries_comparison_{timestamp}.png'), dpi=150)
        plt.close(fig1)

        # Plot 2: Error analysis with statistics
        fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
        
        for i, (axis, label) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
            # Acceleration error
            ax = axes2[0, i]
            ref_data = np.array(self.reference_data[f'accel_{axis}'])
            def_data = np.array(self.defective_data[f'accel_{axis}'])
            min_len = min(len(ref_data), len(def_data))
            error = def_data[:min_len] - ref_data[:min_len]
            timestamps = self.reference_data['timestamps'][:min_len]
            
            ax.plot(timestamps, error, 'b-', alpha=0.7, linewidth=1.0)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(y=np.mean(error), color='r', linestyle='--', label=f'Mean: {np.mean(error):.4f}', alpha=0.8)
            ax.axhline(y=np.mean(error) + 3*np.std(error), color='r', linestyle=':', alpha=0.5)
            ax.axhline(y=np.mean(error) - 3*np.std(error), color='r', linestyle=':', alpha=0.5)
            
            # Highlight outliers
            outliers = np.where(np.abs(error - np.mean(error)) > 3*np.std(error))[0]
            if len(outliers) > 0:
                ax.scatter(timestamps[outliers], error[outliers], color='red', s=30, zorder=5, label=f'{len(outliers)} outliers')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Error (m/s²)')
            ax.set_title(f'Accel {label} Error')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Gyroscope error
            ax = axes2[1, i]
            ref_data = np.array(self.reference_data[f'gyro_{axis}'])
            def_data = np.array(self.defective_data[f'gyro_{axis}'])
            min_len = min(len(ref_data), len(def_data))
            error = def_data[:min_len] - ref_data[:min_len]
            
            ax.plot(timestamps, error, 'r-', alpha=0.7, linewidth=1.0)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(y=np.mean(error), color='b', linestyle='--', label=f'Mean: {np.mean(error):.4f}', alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Error (rad/s)')
            ax.set_title(f'Gyro {label} Error')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        fig2.savefig(os.path.join(self.output_dir, f'02_error_analysis_{timestamp}.png'), dpi=150)
        plt.close(fig2)

        # Plot 3: Distribution histograms with statistics
        fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
        
        for i, (axis, label) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
            # Acceleration distribution
            ax = axes3[0, i]
            ref_data = np.array(self.reference_data[f'accel_{axis}'])
            def_data = np.array(self.defective_data[f'accel_{axis}'])
            
            ax.hist(ref_data, bins=50, alpha=0.5, label=f'Reference', density=True, color='green')
            ax.hist(def_data, bins=50, alpha=0.5, label=f'Defective', density=True, color='red')
            
            # Add vertical lines for statistics
            ref_mean = np.mean(ref_data)
            def_mean = np.mean(def_data)
            ax.axvline(ref_mean, color='g', linestyle='--', alpha=0.7, label=f'Ref Mean: {ref_mean:.3f}')
            ax.axvline(def_mean, color='r', linestyle='--', alpha=0.7, label=f'Def Mean: {def_mean:.3f}')
            
            ax.set_xlabel('Acceleration (m/s²)')
            ax.set_ylabel('Density')
            ax.set_title(f'Accel {label} Distribution')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Gyroscope distribution
            ax = axes3[1, i]
            ref_data = np.array(self.reference_data[f'gyro_{axis}'])
            def_data = np.array(self.defective_data[f'gyro_{axis}'])
            
            ax.hist(ref_data, bins=50, alpha=0.5, label='Reference', density=True, color='blue')
            ax.hist(def_data, bins=50, alpha=0.5, label='Defective', density=True, color='red')
            
            ax.set_xlabel('Angular Velocity (rad/s)')
            ax.set_ylabel('Density')
            ax.set_title(f'Gyro {label} Distribution')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        fig3.savefig(os.path.join(self.output_dir, f'03_distributions_{timestamp}.png'), dpi=150)
        plt.close(fig3)

        # Plot 4: Frequency and timing analysis
        fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
        
        # Frequency histogram
        ax = axes4[0, 0]
        ref_dt = np.array(self.ref_dt_list)
        def_dt = np.array(self.def_dt_list)
        
        if len(ref_dt) > 0:
            ax.hist(1/ref_dt, bins=30, alpha=0.5, label='Reference', density=True, color='green')
        if len(def_dt) > 0:
            ax.hist(1/def_dt, bins=30, alpha=0.5, label='Defective', density=True, color='red')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Density')
        ax.set_title('Message Frequency Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Timing jitter
        ax = axes4[0, 1]
        if len(ref_dt) > 0:
            ax.plot(self.reference_data['timestamps'][1:], ref_dt*1000, 'g.', alpha=0.5, label='Reference', markersize=2)
        if len(def_dt) > 0:
            ax.plot(self.defective_data['timestamps'][1:], def_dt*1000, 'r.', alpha=0.5, label='Defective', markersize=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DT (ms)')
        ax.set_title('Timing Jitter')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Signal power spectrum
        ax = axes4[1, 0]
        for name, color in [('reference', 'green'), ('defective', 'red')]:
            dataset = self.reference_data if name == 'reference' else self.defective_data
            accel_mag = np.sqrt(np.array(dataset['accel_x'])**2 + 
                               np.array(dataset['accel_y'])**2 + 
                               np.array(dataset['accel_z'])**2)
            if len(accel_mag) > 10:
                freqs, psd = periodogram(accel_mag, fs=100, nperseg=min(256, len(accel_mag)//2))
                ax.semilogy(freqs[1:], psd[1:], color=color, alpha=0.7, label=name.capitalize())
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title('Acceleration Power Spectrum')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Correlation heatmap
        ax = axes4[1, 1]
        signals = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        ref_matrix = np.column_stack([self.reference_data[s] for s in signals])
        def_matrix = np.column_stack([self.defective_data[s] for s in signals])
        
        if len(ref_matrix) > 0:
            ref_corr = np.corrcoef(ref_matrix.T)
            im = ax.imshow(ref_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels([s.replace('_', ' ').upper() for s in signals], rotation=45, ha='right')
            ax.set_yticklabels([s.replace('_', ' ').upper() for s in signals])
            ax.set_title('Reference Correlation Matrix')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        fig4.savefig(os.path.join(self.output_dir, f'04_frequency_analysis_{timestamp}.png'), dpi=150)
        plt.close(fig4)

        # Plot 5: Static bias analysis
        fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (name, color) in enumerate([('reference', 'green'), ('defective', 'red')]):
            ax = axes5[idx]
            if name in self.static_bias and 'error' not in self.static_bias[name]:
                bias = self.static_bias[name]
                accel_bias = np.array(bias['accel_bias'])
                gyro_bias = np.array(bias['gyro_bias'])
                
                # Plot accel and gyro bias bars
                x = np.arange(6)
                values = np.concatenate([accel_bias, gyro_bias])
                colors = ['g', 'g', 'g', 'b', 'b', 'b']
                ax.bar(x, values, color=colors, alpha=0.7)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.set_xticks(x)
                ax.set_xticklabels(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
                ax.set_ylabel('Bias')
                ax.set_title(f'{name.capitalize()} Static Bias')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig5.savefig(os.path.join(self.output_dir, f'05_static_bias_{timestamp}.png'), dpi=150)
        plt.close(fig5)

        self.get_logger().info(f"All plots saved to {self.output_dir}")

    def _print_summary(self) -> None:
        """Print a summary of key findings."""
        self.get_logger().info("\n" + "=" * 80)
        self.get_logger().info("ANALYSIS SUMMARY")
        self.get_logger().info("=" * 80)
        
        # Sample counts
        ref_samples = len(self.reference_data['timestamps'])
        def_samples = len(self.defective_data['timestamps'])
        self.get_logger().info(f"Reference samples: {ref_samples}")
        self.get_logger().info(f"Defective samples: {def_samples}")
        
        # Frequency
        if 'reference' in self.frequency_stats:
            freq = self.frequency_stats['reference']
            self.get_logger().info(f"Reference frequency: {freq['mean_freq']:.1f} Hz, jitter: {freq['jitter']:.1f} ms")
        if 'defective' in self.frequency_stats:
            freq = self.frequency_stats['defective']
            self.get_logger().info(f"Defective frequency: {freq['mean_freq']:.1f} Hz, jitter: {freq['jitter']:.1f} ms")
        
        # Static bias
        for name in ['reference', 'defective']:
            if name in self.static_bias and 'error' not in self.static_bias[name]:
                bias = self.static_bias[name]
                accel = bias['accel_bias']
                gyro = bias['gyro_bias']
                self.get_logger().info(f"{name.capitalize()} static bias - Accel: [{accel[0]:.3f}, {accel[1]:.3f}, {accel[2]:.3f}] m/s²")
                self.get_logger().info(f"                        - Gyro:  [{gyro[0]:.3f}, {gyro[1]:.3f}, {gyro[2]:.3f}] rad/s")
        
        # Outliers
        for name in ['reference', 'defective']:
            if name in self.outlier_stats:
                stats = self.outlier_stats[name]
                total = sum([s['total_outliers'] for s in stats.values()])
                total_samples = sum([self.reference_stats.get(k, {}).get('count', 0) for k in stats.keys()])
                self.get_logger().info(f"{name.capitalize()} outliers: {total} ({100*total/total_samples:.1f}%)")
        
        # Frame ID
        if 'reference_frame_id' in self.comparison_stats:
            frame_info = self.comparison_stats['reference_frame_id']
            self.get_logger().info(f"Reference frame_id: {frame_info['most_common']} (consistency: {frame_info['consistency']:.1f}%)")
        if 'defective_frame_id' in self.comparison_stats:
            frame_info = self.comparison_stats['defective_frame_id']
            self.get_logger().info(f"Defective frame_id: {frame_info['most_common']} (consistency: {frame_info['consistency']:.1f}%)")
        
        self.get_logger().info("=" * 80)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IMUDataLogger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()