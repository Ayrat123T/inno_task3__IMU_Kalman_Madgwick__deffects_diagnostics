#!/usr/bin/env python3
"""IMU Data Logger and Comparison Node.

Subscribes: /ouster/imu (reference)
            /ouster/imu_defective (defective)
Publishes:  Nothing

Records IMU data for 60 seconds and generates comparison plots.
"""

from __future__ import annotations

import os
import sys
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Imu
import numpy as np

# Try to import matplotlib, but continue if not available
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
        self.declare_parameter("record_duration", 60.0)  # seconds
        self.declare_parameter("reference_topic", "/ouster/imu")
        self.declare_parameter("defective_topic", "/ouster/imu_defective")
        self.declare_parameter("output_dir", str(os.path.expanduser("~/imu_analysis")))
        self.declare_parameter("plot_covariance", False)
        self.declare_parameter("wait_for_topics", 10.0)  # seconds to wait for topics

        self.record_duration = float(self.get_parameter("record_duration").value)
        self.reference_topic = str(self.get_parameter("reference_topic").value)
        self.defective_topic = str(self.get_parameter("defective_topic").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.plot_covariance = bool(self.get_parameter("plot_covariance").value)
        self.wait_for_topics = float(self.get_parameter("wait_for_topics").value)

        # Data storage
        self.reference_data = {
            'timestamps': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
            'orientation_w': [], 'orientation_x': [], 'orientation_y': [], 'orientation_z': [],
            'accel_cov': [], 'gyro_cov': [], 'orient_cov': []
        }
        
        self.defective_data = {
            'timestamps': [],
            'accel_x': [], 'accel_y': [], 'accel_z': [],
            'gyro_x': [], 'gyro_y': [], 'gyro_z': [],
            'orientation_w': [], 'orientation_x': [], 'orientation_y': [], 'orientation_z': [],
            'accel_cov': [], 'gyro_cov': [], 'orient_cov': []
        }

        # Statistics
        self.reference_stats = {}
        self.defective_stats = {}
        self.comparison_stats = {}

        # Subscribers
        self.ref_sub = self.create_subscription(
            Imu, self.reference_topic, self._reference_callback, 10
        )
        self.def_sub = self.create_subscription(
            Imu, self.defective_topic, self._defective_callback, 10
        )

        # State variables
        self.start_time = None
        self.recording_active = False
        self.recording_started = False
        self.ref_msg_count = 0
        self.def_msg_count = 0
        
        # Check if topics are available
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"IMU Data Logger started")
        self.get_logger().info(f"Recording duration: {self.record_duration} seconds")
        self.get_logger().info(f"Reference topic: {self.reference_topic}")
        self.get_logger().info(f"Defective topic: {self.defective_topic}")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("=" * 60)
        
        # Create timer to check for topic availability
        self.check_timer = self.create_timer(0.5, self._check_topics)
        self.get_logger().info("Waiting for topics to become available...")

    def _check_topics(self) -> None:
        """Check if topics are available and start recording."""
        if self.recording_started:
            return
            
        # Check if we've been waiting too long
        if not hasattr(self, '_check_start_time'):
            self._check_start_time = time.time()
        
        elapsed = time.time() - self._check_start_time
        if elapsed > self.wait_for_topics:
            self.get_logger().warn(f"Timeout waiting for topics after {self.wait_for_topics} seconds. Exiting...")
            rclpy.shutdown()
            sys.exit(0)
        
        # Check if topics have subscribers (just check if we've received any messages)
        if self.ref_msg_count > 0 and self.def_msg_count > 0:
            self.get_logger().info(f"Both topics are active! Reference: {self.ref_msg_count} msgs, Defective: {self.def_msg_count} msgs")
            self.get_logger().info("Starting recording in 5 seconds...")
            self.check_timer.cancel()
            
            # Schedule recording start
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
        
        # Create timer to stop recording
        self.stop_timer = self.create_timer(self.record_duration, self._stop_recording)

    def _stop_recording(self) -> None:
        """Stop recording and analyze data."""
        if not self.recording_active:
            return
        
        self.recording_active = False
        self.stop_timer.cancel()
        self.get_logger().info(f"Recording complete! Collected {len(self.reference_data['timestamps'])} reference samples and {len(self.defective_data['timestamps'])} defective samples.")
        
        # Check if we have enough data
        if len(self.reference_data['timestamps']) < 10 or len(self.defective_data['timestamps']) < 10:
            self.get_logger().error("Not enough data collected! Check that topics are publishing messages.")
            self.get_logger().info("Exiting...")
            rclpy.shutdown()
            sys.exit(0)
        
        self._analyze_data()
        self._generate_plots()
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
        
        # Store data
        self.reference_data['timestamps'].append(rel_time)
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
        
        # Covariances (diagonal elements)
        self.reference_data['accel_cov'].append([
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[8]
        ])
        self.reference_data['gyro_cov'].append([
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[8]
        ])
        self.reference_data['orient_cov'].append([
            msg.orientation_covariance[0],
            msg.orientation_covariance[4],
            msg.orientation_covariance[8]
        ])

    def _defective_callback(self, msg: Imu) -> None:
        """Callback for defective IMU data."""
        self.def_msg_count += 1
        
        if not self.recording_active:
            return
        
        timestamp = self._get_seconds(msg.header.stamp)
        rel_time = timestamp - self.start_time
        
        # Store data
        self.defective_data['timestamps'].append(rel_time)
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
        
        # Covariances (diagonal elements)
        self.defective_data['accel_cov'].append([
            msg.linear_acceleration_covariance[0],
            msg.linear_acceleration_covariance[4],
            msg.linear_acceleration_covariance[8]
        ])
        self.defective_data['gyro_cov'].append([
            msg.angular_velocity_covariance[0],
            msg.angular_velocity_covariance[4],
            msg.angular_velocity_covariance[8]
        ])
        self.defective_data['orient_cov'].append([
            msg.orientation_covariance[0],
            msg.orientation_covariance[4],
            msg.orientation_covariance[8]
        ])

    def _get_seconds(self, stamp) -> float:
        """Convert ROS timestamp to seconds."""
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _analyze_data(self) -> None:
        """Compute statistics for both datasets."""
        self.get_logger().info("Analyzing data...")
        
        # Compute statistics for reference data
        for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            data = np.array(self.reference_data[key])
            if len(data) > 0:
                self.reference_stats[key] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'rms': np.sqrt(np.mean(data**2))
                }
        
        # Compute statistics for defective data
        for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            data = np.array(self.defective_data[key])
            if len(data) > 0:
                self.defective_stats[key] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'rms': np.sqrt(np.mean(data**2))
                }
        
        # Comparison statistics
        for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            if key in self.reference_stats and key in self.defective_stats:
                ref_mean = self.reference_stats[key]['mean']
                def_mean = self.defective_stats[key]['mean']
                ref_std = self.reference_stats[key]['std']
                def_std = self.defective_stats[key]['std']
                
                # Align data lengths
                min_len = min(len(self.reference_data[key]), len(self.defective_data[key]))
                ref_data = np.array(self.reference_data[key][:min_len])
                def_data = np.array(self.defective_data[key][:min_len])
                
                self.comparison_stats[key] = {
                    'bias': def_mean - ref_mean,
                    'std_ratio': def_std / ref_std if ref_std > 0 else float('inf'),
                    'error_rms': np.sqrt(np.mean((ref_data - def_data)**2)),
                    'max_error': np.max(np.abs(ref_data - def_data)),
                    'mean_error': np.mean(ref_data - def_data)
                }

        # Print statistics
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("STATISTICS SUMMARY")
        self.get_logger().info("=" * 70)
        
        self.get_logger().info("ACCELEROMETER:")
        for key in ['accel_x', 'accel_y', 'accel_z']:
            if key in self.comparison_stats:
                stats = self.comparison_stats[key]
                self.get_logger().info(
                    f"  {key:10s} | Bias: {stats['bias']:8.3f} | Std Ratio: {stats['std_ratio']:6.2f} | RMS Error: {stats['error_rms']:8.3f}"
                )
        
        self.get_logger().info("\nGYROSCOPE:")
        for key in ['gyro_x', 'gyro_y', 'gyro_z']:
            if key in self.comparison_stats:
                stats = self.comparison_stats[key]
                self.get_logger().info(
                    f"  {key:10s} | Bias: {stats['bias']:8.3f} | Std Ratio: {stats['std_ratio']:6.2f} | RMS Error: {stats['error_rms']:8.3f}"
                )
        self.get_logger().info("=" * 70)

    def _generate_plots(self) -> None:
        """Generate comparison plots."""
        if not MATPLOTLIB_AVAILABLE:
            self.get_logger().error("Matplotlib not available. Install with: pip install matplotlib")
            return
            
        self.get_logger().info("Generating plots...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if we have data
        if len(self.reference_data['timestamps']) == 0 or len(self.defective_data['timestamps']) == 0:
            self.get_logger().error("No data to plot!")
            return
        
        try:
            # Configure plot style
            rcParams['font.size'] = 10
            rcParams['figure.figsize'] = (12, 8)
            
            # 1. Linear Acceleration Comparison (3 axes)
            fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10))
            
            for i, (axis, label) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
                ref_key = f'accel_{axis}'
                def_key = f'accel_{axis}'
                
                # Align data lengths
                min_len = min(len(self.reference_data['timestamps']), len(self.defective_data['timestamps']))
                timestamps_ref = self.reference_data['timestamps'][:min_len]
                timestamps_def = self.defective_data['timestamps'][:min_len]
                
                axes1[i].plot(timestamps_ref, self.reference_data[ref_key][:min_len], 
                             label=f'Reference Accel {label}', alpha=0.7, linewidth=1.5)
                axes1[i].plot(timestamps_def, self.defective_data[def_key][:min_len], 
                             label=f'Defective Accel {label}', alpha=0.7, linewidth=1.5)
                
                axes1[i].set_xlabel('Time (s)')
                axes1[i].set_ylabel(f'Acceleration (m/s²)')
                axes1[i].set_title(f'Acceleration - {label} Axis')
                axes1[i].grid(True, alpha=0.3)
                axes1[i].legend()
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, f'acceleration_comparison_{timestamp}.png')
            fig1.savefig(output_file, dpi=150)
            plt.close(fig1)
            self.get_logger().info(f"Saved acceleration plot to {output_file}")

            # 2. Angular Velocity Comparison (3 axes)
            fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
            
            for i, (axis, label) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
                ref_key = f'gyro_{axis}'
                def_key = f'gyro_{axis}'
                
                min_len = min(len(self.reference_data['timestamps']), len(self.defective_data['timestamps']))
                timestamps_ref = self.reference_data['timestamps'][:min_len]
                timestamps_def = self.defective_data['timestamps'][:min_len]
                
                axes2[i].plot(timestamps_ref, self.reference_data[ref_key][:min_len], 
                             label=f'Reference Gyro {label}', alpha=0.7, linewidth=1.5)
                axes2[i].plot(timestamps_def, self.defective_data[def_key][:min_len], 
                             label=f'Defective Gyro {label}', alpha=0.7, linewidth=1.5)
                
                axes2[i].set_xlabel('Time (s)')
                axes2[i].set_ylabel('Angular Velocity (rad/s)')
                axes2[i].set_title(f'Angular Velocity - {label} Axis')
                axes2[i].grid(True, alpha=0.3)
                axes2[i].legend()
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, f'gyroscope_comparison_{timestamp}.png')
            fig2.savefig(output_file, dpi=150)
            plt.close(fig2)
            self.get_logger().info(f"Saved gyroscope plot to {output_file}")

            # 3. Error Analysis Plots
            fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
            
            for i, (axis, label) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
                # Acceleration errors
                ref_acc = np.array(self.reference_data[f'accel_{axis}'])
                def_acc = np.array(self.defective_data[f'accel_{axis}'])
                min_len = min(len(ref_acc), len(def_acc))
                error_acc = def_acc[:min_len] - ref_acc[:min_len]
                timestamps = self.reference_data['timestamps'][:min_len]
                
                axes3[0, i].plot(timestamps, error_acc, 'b-', alpha=0.7, linewidth=1.0)
                axes3[0, i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes3[0, i].axhline(y=np.mean(error_acc), color='r', linestyle='-', 
                                   label=f'Mean: {np.mean(error_acc):.3f}', alpha=0.7)
                axes3[0, i].set_xlabel('Time (s)')
                axes3[0, i].set_ylabel('Error (m/s²)')
                axes3[0, i].set_title(f'Accel {label} - Error')
                axes3[0, i].grid(True, alpha=0.3)
                axes3[0, i].legend()
                
                # Gyroscope errors
                ref_gyro = np.array(self.reference_data[f'gyro_{axis}'])
                def_gyro = np.array(self.defective_data[f'gyro_{axis}'])
                min_len = min(len(ref_gyro), len(def_gyro))
                error_gyro = def_gyro[:min_len] - ref_gyro[:min_len]
                
                axes3[1, i].plot(timestamps, error_gyro, 'r-', alpha=0.7, linewidth=1.0)
                axes3[1, i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes3[1, i].axhline(y=np.mean(error_gyro), color='b', linestyle='-',
                                   label=f'Mean: {np.mean(error_gyro):.3f}', alpha=0.7)
                axes3[1, i].set_xlabel('Time (s)')
                axes3[1, i].set_ylabel('Error (rad/s)')
                axes3[1, i].set_title(f'Gyro {label} - Error')
                axes3[1, i].grid(True, alpha=0.3)
                axes3[1, i].legend()
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, f'error_analysis_{timestamp}.png')
            fig3.savefig(output_file, dpi=150)
            plt.close(fig3)
            self.get_logger().info(f"Saved error analysis plot to {output_file}")

            # 4. Histograms - Distribution Comparison
            fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))
            
            for i, (axis, label) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
                # Acceleration histograms
                ref_acc = np.array(self.reference_data[f'accel_{axis}'])
                def_acc = np.array(self.defective_data[f'accel_{axis}'])
                
                axes4[0, i].hist(ref_acc, bins=50, alpha=0.5, label='Reference', density=True)
                axes4[0, i].hist(def_acc, bins=50, alpha=0.5, label='Defective', density=True)
                axes4[0, i].set_xlabel('Acceleration (m/s²)')
                axes4[0, i].set_ylabel('Density')
                axes4[0, i].set_title(f'Accel {label} Distribution')
                axes4[0, i].legend()
                axes4[0, i].grid(True, alpha=0.3)
                
                # Gyroscope histograms
                ref_gyro = np.array(self.reference_data[f'gyro_{axis}'])
                def_gyro = np.array(self.defective_data[f'gyro_{axis}'])
                
                axes4[1, i].hist(ref_gyro, bins=50, alpha=0.5, label='Reference', density=True)
                axes4[1, i].hist(def_gyro, bins=50, alpha=0.5, label='Defective', density=True)
                axes4[1, i].set_xlabel('Angular Velocity (rad/s)')
                axes4[1, i].set_ylabel('Density')
                axes4[1, i].set_title(f'Gyro {label} Distribution')
                axes4[1, i].legend()
                axes4[1, i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, f'histogram_comparison_{timestamp}.png')
            fig4.savefig(output_file, dpi=150)
            plt.close(fig4)
            self.get_logger().info(f"Saved histogram plot to {output_file}")

            # 5. Orientation Comparison
            fig5, axes5 = plt.subplots(4, 1, figsize=(12, 12))
            
            for i, (key, label) in enumerate([('orientation_w', 'W'), ('orientation_x', 'X'), 
                                             ('orientation_y', 'Y'), ('orientation_z', 'Z')]):
                min_len = min(len(self.reference_data['timestamps']), len(self.defective_data['timestamps']))
                timestamps_ref = self.reference_data['timestamps'][:min_len]
                timestamps_def = self.defective_data['timestamps'][:min_len]
                
                axes5[i].plot(timestamps_ref, self.reference_data[key][:min_len], 
                             label=f'Reference {label}', alpha=0.7, linewidth=1.5)
                axes5[i].plot(timestamps_def, self.defective_data[key][:min_len], 
                             label=f'Defective {label}', alpha=0.7, linewidth=1.5)
                axes5[i].set_xlabel('Time (s)')
                axes5[i].set_ylabel('Quaternion Component')
                axes5[i].set_title(f'Orientation - {label}')
                axes5[i].grid(True, alpha=0.3)
                axes5[i].legend()
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, f'orientation_comparison_{timestamp}.png')
            fig5.savefig(output_file, dpi=150)
            plt.close(fig5)
            self.get_logger().info(f"Saved orientation plot to {output_file}")

            # Save statistics to file
            stats_file = os.path.join(self.output_dir, f'statistics_{timestamp}.txt')
            with open(stats_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("IMU DATA ANALYSIS - STATISTICS SUMMARY\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("REFERENCE DATA STATISTICS:\n")
                f.write("-" * 50 + "\n")
                for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                    if key in self.reference_stats:
                        stats = self.reference_stats[key]
                        f.write(f"{key:10s} | Mean: {stats['mean']:8.3f} | Std: {stats['std']:8.3f} | RMS: {stats['rms']:8.3f}\n")
                
                f.write("\nDEFECTIVE DATA STATISTICS:\n")
                f.write("-" * 50 + "\n")
                for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                    if key in self.defective_stats:
                        stats = self.defective_stats[key]
                        f.write(f"{key:10s} | Mean: {stats['mean']:8.3f} | Std: {stats['std']:8.3f} | RMS: {stats['rms']:8.3f}\n")
                
                f.write("\nCOMPARISON STATISTICS:\n")
                f.write("-" * 50 + "\n")
                for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                    if key in self.comparison_stats:
                        stats = self.comparison_stats[key]
                        f.write(f"{key:10s} | Bias: {stats['bias']:8.3f} | Std Ratio: {stats['std_ratio']:6.2f} | RMS Error: {stats['error_rms']:8.3f}\n")
                
                f.write("\n" + "=" * 70 + "\n")
            
            self.get_logger().info(f"Saved statistics to {stats_file}")
            
        except Exception as e:
            self.get_logger().error(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()


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