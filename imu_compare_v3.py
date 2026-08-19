#!/usr/bin/env python3
"""IMU Comparison Tool - Dual Analysis (Clean vs Defective).

Subscribes:
  Clean topics:
    - /ouster/imu (raw clean)
    - /imu/kalman_1d_clean (kalman filtered clean)
    - /imu/madgwick_clean (madgwick filtered clean)
  Defective topics:
    - /ouster/imu_defective (raw defective)
    - /imu/kalman_1d_defective (kalman filtered defective)
    - /imu/madgwick_defective (madgwick filtered defective)

Analyzes and compares both clean and defective data streams.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import csv
import os
from datetime import datetime


class IMUComparator(Node):
    def __init__(self):
        super().__init__('imu_comparator_dual')

        # Clean topics
        self.sub_raw_clean = self.create_subscription(Imu, '/ouster/imu', self.raw_clean_callback, 10)
        self.sub_kalman_clean = self.create_subscription(Imu, '/imu/kalman_1d_clean', self.kalman_clean_callback, 10)
        self.sub_madgwick_clean = self.create_subscription(Imu, '/imu/madgwick_clean', self.madgwick_clean_callback, 10)

        # Defective topics
        self.sub_raw_defective = self.create_subscription(Imu, '/ouster/imu_defective', self.raw_defective_callback, 10)
        self.sub_kalman_defective = self.create_subscription(Imu, '/imu/kalman_1d_defective', self.kalman_defective_callback, 10)
        self.sub_madgwick_defective = self.create_subscription(Imu, '/imu/madgwick_defective', self.madgwick_defective_callback, 10)

        # Data storage: (timestamp, roll, pitch, yaw)
        self.data = {
            'clean': {
                'raw': [],
                'kalman': [],
                'madgwick': []
            },
            'defective': {
                'raw': [],
                'kalman': [],
                'madgwick': []
            }
        }

        # Message counters
        self.counts = {
            'clean': {'raw': 0, 'kalman': 0, 'madgwick': 0},
            'defective': {'raw': 0, 'kalman': 0, 'madgwick': 0}
        }

        # Collection parameters
        self.duration = 60.0  # seconds
        self.start_time = time.time()
        self.collecting = True
        self.analysis_done = False

        # Timer for collection completion
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("=" * 70)
        self.get_logger().info("IMU Dual Comparator Started")
        self.get_logger().info(f"Collecting data for {self.duration} seconds")
        self.get_logger().info("")
        self.get_logger().info("Clean topics:")
        self.get_logger().info("  - /ouster/imu (raw)")
        self.get_logger().info("  - /imu/kalman_1d_clean (Kalman)")
        self.get_logger().info("  - /imu/madgwick_clean (Madgwick)")
        self.get_logger().info("")
        self.get_logger().info("Defective topics:")
        self.get_logger().info("  - /ouster/imu_defective (raw)")
        self.get_logger().info("  - /imu/kalman_1d_defective (Kalman)")
        self.get_logger().info("  - /imu/madgwick_defective (Madgwick)")
        self.get_logger().info("=" * 70)

    def imu_to_euler(self, msg):
        """Extract Euler angles (roll, pitch, yaw) from IMU message."""
        q = msg.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        return r.as_euler('xyz')  # radians

    def get_timestamp(self, msg):
        """Extract timestamp from IMU message."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def store_data(self, source, filter_type, msg):
        """Store data for a specific source and filter type."""
        if not self.collecting or self.analysis_done:
            return
        
        t = self.get_timestamp(msg)
        roll, pitch, yaw = self.imu_to_euler(msg)
        self.data[source][filter_type].append((t, roll, pitch, yaw))
        self.counts[source][filter_type] += 1

    # Clean callbacks
    def raw_clean_callback(self, msg):
        self.store_data('clean', 'raw', msg)

    def kalman_clean_callback(self, msg):
        self.store_data('clean', 'kalman', msg)

    def madgwick_clean_callback(self, msg):
        self.store_data('clean', 'madgwick', msg)

    # Defective callbacks
    def raw_defective_callback(self, msg):
        self.store_data('defective', 'raw', msg)

    def kalman_defective_callback(self, msg):
        self.store_data('defective', 'kalman', msg)

    def madgwick_defective_callback(self, msg):
        self.store_data('defective', 'madgwick', msg)

    def timer_callback(self):
        """Check if collection duration has elapsed."""
        if self.analysis_done:
            return
        if self.collecting and (time.time() - self.start_time) >= self.duration:
            self.collecting = False
            self.get_logger().info('Data collection finished, analyzing...')
            self.analyze_and_plot()

    def analyze_and_plot(self):
        """Perform comprehensive analysis and generate plots."""
        if self.analysis_done:
            return
        self.analysis_done = True

        # Convert data to arrays
        clean_data = {}
        defective_data = {}
        
        for filter_type in ['raw', 'kalman', 'madgwick']:
            clean_data[filter_type] = np.array(self.data['clean'][filter_type]) if self.data['clean'][filter_type] else np.empty((0, 4))
            defective_data[filter_type] = np.array(self.data['defective'][filter_type]) if self.data['defective'][filter_type] else np.empty((0, 4))

        # Print message counts
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("DATA COLLECTION SUMMARY")
        self.get_logger().info("=" * 70)
        
        for source in ['clean', 'defective']:
            self.get_logger().info(f"{source.upper()} stream:")
            for filter_type, count in self.counts[source].items():
                self.get_logger().info(f"  {filter_type:10s}: {count:4d} messages")
        
        # Check if we have data
        total_clean = sum([len(clean_data[f]) for f in ['raw', 'kalman', 'madgwick']])
        total_defective = sum([len(defective_data[f]) for f in ['raw', 'kalman', 'madgwick']])
        
        if total_clean == 0 and total_defective == 0:
            self.get_logger().error('No data received from any topic!')
            return

        # Analyze each stream
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("STATISTICAL ANALYSIS")
        self.get_logger().info("=" * 70)

        analysis_results = {}
        
        for source_name, source_data in [('CLEAN', clean_data), ('DEFECTIVE', defective_data)]:
            self.get_logger().info(f"\n{source_name} STREAM:")
            self.get_logger().info("-" * 50)  # Исправлено: добавлены скобки
            
            for filter_type, data in source_data.items():
                if len(data) == 0:
                    self.get_logger().info(f"  {filter_type.upper()}: No data")
                    continue
                
                # Calculate metrics
                metrics = self.compute_metrics(data)
                analysis_results[f"{source_name}_{filter_type}"] = metrics
                
                self.get_logger().info(f"  {filter_type.upper()}:")
                self.get_logger().info(f"    Noise (std) [rad]:  roll={metrics['noise'][0]:.4f}, pitch={metrics['noise'][1]:.4f}, yaw={metrics['noise'][2]:.4f}")
                self.get_logger().info(f"    Drift [rad/s]:       roll={metrics['drift'][0]:.4e}, pitch={metrics['drift'][1]:.4e}, yaw={metrics['drift'][2]:.4e}")
                self.get_logger().info(f"    Settling time [s]:   {metrics['settling']:.2f}")
                self.get_logger().info(f"    Mean [rad]:          roll={metrics['mean'][0]:.4f}, pitch={metrics['mean'][1]:.4f}, yaw={metrics['mean'][2]:.4f}")
                self.get_logger().info(f"    Std [rad]:           roll={metrics['std'][0]:.4f}, pitch={metrics['std'][1]:.4f}, yaw={metrics['std'][2]:.4f}")

        # Save data to CSV
        self.save_to_csv(clean_data, defective_data)

        # Generate plots
        self.plot_comparison(clean_data, defective_data)

    def compute_metrics(self, data):
        """Compute comprehensive metrics for a dataset."""
        if len(data) < 5:
            return {'noise': np.full(3, np.nan), 'drift': np.full(3, np.nan), 
                    'settling': np.nan, 'mean': np.full(3, np.nan), 'std': np.full(3, np.nan)}
        
        # Static noise (first 10% of data)
        n_static = max(2, int(len(data) * 0.1))
        static = data[:n_static, 1:]
        noise = np.std(static, axis=0)
        
        # Drift (linear trend)
        t = data[:, 0] - data[0, 0]
        angles = data[:, 1:]
        drifts = []
        for i in range(3):
            A = np.vstack([t, np.ones(len(t))]).T
            slope, _ = np.linalg.lstsq(A, angles[:, i], rcond=None)[0]
            drifts.append(slope)
        drift = np.array(drifts)
        
        # Settling time
        settling = self.estimate_settling_time(data)
        
        # Overall statistics
        mean = np.mean(angles, axis=0)
        std = np.std(angles, axis=0)
        
        return {'noise': noise, 'drift': drift, 'settling': settling, 
                'mean': mean, 'std': std}

    def estimate_settling_time(self, data, threshold_std=2.0):
        """Estimate settling time based on final steady-state."""
        n = len(data)
        if n < 20:
            return np.nan
        
        n_final = max(5, int(n * 0.2))
        final = data[-n_final:, 1:]
        final_mean = np.mean(final, axis=0)
        final_std = np.std(final, axis=0)
        
        if np.any(final_std < 1e-12):
            return np.nan
        
        in_band = np.abs(data[:, 1:] - final_mean) < threshold_std * final_std
        all_in_band = np.all(in_band, axis=1)
        
        start_idx = 0
        for i in range(n):
            if np.all(all_in_band[i:]):
                start_idx = i
                break
        
        if start_idx == 0 and not np.all(all_in_band):
            return np.nan
        
        return data[start_idx, 0] - data[0, 0]

    def save_to_csv(self, clean_data, defective_data):
        """Save all data to CSV with dual streams."""
        # Determine max length for each stream
        max_len_clean = max([len(clean_data[f]) for f in ['raw', 'kalman', 'madgwick']])
        max_len_defective = max([len(defective_data[f]) for f in ['raw', 'kalman', 'madgwick']])
        max_len = max(max_len_clean, max_len_defective)
        
        if max_len == 0:
            return

        # Pad data to equal length
        def pad_data(data, length):
            if len(data) == 0:
                return np.full((length, 4), np.nan)
            if len(data) < length:
                pad_rows = length - len(data)
                return np.vstack([data, np.full((pad_rows, 4), np.nan)])
            return data

        clean_padded = {}
        defective_padded = {}
        
        for filter_type in ['raw', 'kalman', 'madgwick']:
            clean_padded[filter_type] = pad_data(clean_data[filter_type], max_len)
            defective_padded[filter_type] = pad_data(defective_data[filter_type], max_len)

        # Create headers
        header = []
        for source in ['clean', 'defective']:
            for filter_type in ['raw', 'kalman', 'madgwick']:
                header.extend([
                    f'time_{source}_{filter_type}',
                    f'roll_{source}_{filter_type}',
                    f'pitch_{source}_{filter_type}',
                    f'yaw_{source}_{filter_type}'
                ])

        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'imu_dual_comparison_{timestamp}.csv'
        filepath = os.path.join(os.getcwd(), filename)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i in range(max_len):
                row = []
                for source in ['clean', 'defective']:
                    for filter_type in ['raw', 'kalman', 'madgwick']:
                        data = clean_padded if source == 'clean' else defective_padded
                        row.extend([
                            data[filter_type][i, 0],
                            data[filter_type][i, 1],
                            data[filter_type][i, 2],
                            data[filter_type][i, 3]
                        ])
                writer.writerow(row)

        self.get_logger().info(f"\nData saved to: {filepath}")

    def plot_comparison(self, clean_data, defective_data):
        """Generate comprehensive comparison plots."""
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('IMU Comparison: Clean vs Defective Streams', fontsize=16)

        # Colors and styles
        colors = {'raw': '#1f77b4', 'kalman': '#ff7f0e', 'madgwick': '#2ca02c'}
        styles = {'raw': '-', 'kalman': '--', 'madgwick': ':'}
        
        # Check if we have data to plot
        has_clean = any([len(clean_data[f]) > 0 for f in ['raw', 'kalman', 'madgwick']])
        has_defective = any([len(defective_data[f]) > 0 for f in ['raw', 'kalman', 'madgwick']])
        
        if not has_clean and not has_defective:
            self.get_logger().warn('No data to plot')
            plt.close(fig)
            return

        # Plot each angle
        for i, (angle_label, angle_idx) in enumerate([('Roll', 1), ('Pitch', 2), ('Yaw', 3)]):
            # Clean stream
            ax_clean = axes[i, 0]
            if has_clean:
                for filter_type in ['raw', 'kalman', 'madgwick']:
                    data = clean_data[filter_type]
                    if len(data) > 0:
                        t = data[:, 0] - data[0, 0]
                        ax_clean.plot(t, data[:, angle_idx], 
                                    color=colors[filter_type], 
                                    linestyle=styles[filter_type],
                                    label=filter_type.capitalize(), 
                                    alpha=0.8)
                ax_clean.set_title(f'Clean - {angle_label}', fontsize=12)
                ax_clean.set_ylabel('Angle (rad)')
                ax_clean.legend()
                ax_clean.grid(True, alpha=0.3)
            else:
                ax_clean.text(0.5, 0.5, 'No clean data', 
                             ha='center', va='center', transform=ax_clean.transAxes)
                ax_clean.set_title(f'Clean - {angle_label} (No Data)')
            
            # Defective stream
            ax_def = axes[i, 1]
            if has_defective:
                for filter_type in ['raw', 'kalman', 'madgwick']:
                    data = defective_data[filter_type]
                    if len(data) > 0:
                        t = data[:, 0] - data[0, 0]
                        ax_def.plot(t, data[:, angle_idx],
                                    color=colors[filter_type],
                                    linestyle=styles[filter_type],
                                    label=filter_type.capitalize(),
                                    alpha=0.8)
                ax_def.set_title(f'Defective - {angle_label}', fontsize=12)
                ax_def.set_ylabel('Angle (rad)')
                ax_def.legend()
                ax_def.grid(True, alpha=0.3)
            else:
                ax_def.text(0.5, 0.5, 'No defective data',
                           ha='center', va='center', transform=ax_def.transAxes)
                ax_def.set_title(f'Defective - {angle_label} (No Data)')

        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()

        # Additional comparison plot: Raw Clean vs Raw Defective
        if has_clean and has_defective:
            fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10))
            fig2.suptitle('Raw IMU Comparison: Clean vs Defective', fontsize=14)
            
            for i, (angle_label, angle_idx) in enumerate([('Roll', 1), ('Pitch', 2), ('Yaw', 3)]):
                ax = axes2[i]
                
                # Clean raw
                if len(clean_data['raw']) > 0:
                    t = clean_data['raw'][:, 0] - clean_data['raw'][0, 0]
                    ax.plot(t, clean_data['raw'][:, angle_idx], 
                           'b-', label='Clean', alpha=0.7, linewidth=1.5)
                
                # Defective raw
                if len(defective_data['raw']) > 0:
                    t = defective_data['raw'][:, 0] - defective_data['raw'][0, 0]
                    ax.plot(t, defective_data['raw'][:, angle_idx],
                           'r-', label='Defective', alpha=0.7, linewidth=1.5)
                
                ax.set_ylabel(f'{angle_label} (rad)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            axes2[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.show()

        # Additional comparison: Filter Effectiveness
        fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10))
        fig3.suptitle('Filter Effectiveness: Clean vs Defective', fontsize=14)
        
        for i, (angle_label, angle_idx) in enumerate([('Roll', 1), ('Pitch', 2), ('Yaw', 3)]):
            ax = axes3[i]
            
            for filter_type, color, style in [('raw', 'gray', '--'), 
                                              ('kalman', 'blue', '-'), 
                                              ('madgwick', 'green', '-')]:
                # Difference between defective and clean for each filter
                if len(defective_data[filter_type]) > 0 and len(clean_data[filter_type]) > 0:
                    # Align lengths
                    min_len = min(len(defective_data[filter_type]), len(clean_data[filter_type]))
                    diff = defective_data[filter_type][:min_len, angle_idx] - clean_data[filter_type][:min_len, angle_idx]
                    t = clean_data[filter_type][:min_len, 0] - clean_data[filter_type][0, 0]
                    
                    ax.plot(t, diff, 
                           color=color, linestyle=style,
                           label=f'{filter_type.capitalize()} error',
                           alpha=0.7, linewidth=1.2)
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_ylabel(f'{angle_label} Error (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes3[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = IMUComparator()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()