#!/usr/bin/env python3
"""IMU Defect Simulator Node.

Subscribes: /ouster/imu
Publishes:  /ouster/imu_defective

Simulates various IMU defects for testing purposes:
- Zero bias (offset)
- Increased noise
- Axis sign flips
- Frame_id corruption
- Message drops
- Timestamp delays
- Wrong covariance values
- Scale errors
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from rclpy.clock import Clock


class IMUDefectSimulator(Node):
    def __init__(self) -> None:
        super().__init__("imu_defect_simulator")

        # Parameters for defect simulation
        self.declare_parameter("input_topic", "/ouster/imu")
        self.declare_parameter("output_topic", "/ouster/imu_defective")
        self.declare_parameter("enable_zero_bias", True)
        self.declare_parameter("bias_accel", [0.05, 0.03, -0.02])  # m/s²
        self.declare_parameter("bias_gyro", [0.01, -0.015, 0.005])  # rad/s
        
        self.declare_parameter("enable_noise", True)
        self.declare_parameter("noise_accel_std", 0.02)  # m/s²
        self.declare_parameter("noise_gyro_std", 0.005)  # rad/s
        
        self.declare_parameter("enable_axis_flip", True)
        self.declare_parameter("flip_accel_x", False)
        self.declare_parameter("flip_accel_y", True)   # Flip Y axis
        self.declare_parameter("flip_accel_z", False)
        self.declare_parameter("flip_gyro_x", False)
        self.declare_parameter("flip_gyro_y", False)
        self.declare_parameter("flip_gyro_z", True)    # Flip Z axis
        
        self.declare_parameter("enable_drop_messages", True)
        self.declare_parameter("drop_rate", 0.1)  # 10% drop rate
        
        self.declare_parameter("enable_timestamp_delay", True)
        self.declare_parameter("timestamp_delay_sec", 0.05)  # 50ms delay
        
        self.declare_parameter("enable_frame_id_corruption", True)
        self.declare_parameter("corrupt_frame_id", "defective_imu_frame")
        
        self.declare_parameter("enable_covariance_errors", True)
        self.declare_parameter("covariance_scale", 10.0)  # Magnify covariance
        
        self.declare_parameter("enable_scale_errors", True)
        self.declare_parameter("scale_accel", [1.0, 1.05, 0.95])  # 5% error on Y, -5% on Z
        self.declare_parameter("scale_gyro", [1.0, 0.97, 1.03])   # 3% error on Y, 3% on Z

        # Get parameter values
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        
        self.enable_zero_bias = bool(self.get_parameter("enable_zero_bias").value)
        self.bias_accel = list(self.get_parameter("bias_accel").value)
        self.bias_gyro = list(self.get_parameter("bias_gyro").value)
        
        self.enable_noise = bool(self.get_parameter("enable_noise").value)
        self.noise_accel_std = float(self.get_parameter("noise_accel_std").value)
        self.noise_gyro_std = float(self.get_parameter("noise_gyro_std").value)
        
        self.enable_axis_flip = bool(self.get_parameter("enable_axis_flip").value)
        self.flip_accel_x = bool(self.get_parameter("flip_accel_x").value)
        self.flip_accel_y = bool(self.get_parameter("flip_accel_y").value)
        self.flip_accel_z = bool(self.get_parameter("flip_accel_z").value)
        self.flip_gyro_x = bool(self.get_parameter("flip_gyro_x").value)
        self.flip_gyro_y = bool(self.get_parameter("flip_gyro_y").value)
        self.flip_gyro_z = bool(self.get_parameter("flip_gyro_z").value)
        
        self.enable_drop_messages = bool(self.get_parameter("enable_drop_messages").value)
        self.drop_rate = float(self.get_parameter("drop_rate").value)
        
        self.enable_timestamp_delay = bool(self.get_parameter("enable_timestamp_delay").value)
        self.timestamp_delay_sec = float(self.get_parameter("timestamp_delay_sec").value)
        
        self.enable_frame_id_corruption = bool(self.get_parameter("enable_frame_id_corruption").value)
        self.corrupt_frame_id = str(self.get_parameter("corrupt_frame_id").value)
        
        self.enable_covariance_errors = bool(self.get_parameter("enable_covariance_errors").value)
        self.covariance_scale = float(self.get_parameter("covariance_scale").value)
        
        self.enable_scale_errors = bool(self.get_parameter("enable_scale_errors").value)
        self.scale_accel = list(self.get_parameter("scale_accel").value)
        self.scale_gyro = list(self.get_parameter("scale_gyro").value)

        # Publishers and Subscribers
        self.publisher = self.create_publisher(Imu, self.output_topic, 10)
        self.subscription = self.create_subscription(
            Imu, self.input_topic, self._imu_callback, 10
        )

        self.message_count = 0
        
        # Log startup info
        self.get_logger().info("=" * 60)
        self.get_logger().info("IMU Defect Simulator Started")
        self.get_logger().info(f"Input:  {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")
        self.get_logger().info("Active defects:")
        if self.enable_zero_bias:
            self.get_logger().info(f"  - Zero bias: Accel {self.bias_accel}, Gyro {self.bias_gyro}")
        if self.enable_noise:
            self.get_logger().info(f"  - Noise: Accel σ={self.noise_accel_std}, Gyro σ={self.noise_gyro_std}")
        if self.enable_axis_flip:
            flips = []
            if self.flip_accel_x: flips.append("AccelX")
            if self.flip_accel_y: flips.append("AccelY")
            if self.flip_accel_z: flips.append("AccelZ")
            if self.flip_gyro_x: flips.append("GyroX")
            if self.flip_gyro_y: flips.append("GyroY")
            if self.flip_gyro_z: flips.append("GyroZ")
            self.get_logger().info(f"  - Axis flips: {flips}")
        if self.enable_drop_messages:
            self.get_logger().info(f"  - Message drops: {self.drop_rate*100:.0f}%")
        if self.enable_timestamp_delay:
            self.get_logger().info(f"  - Timestamp delay: {self.timestamp_delay_sec*1000:.0f}ms")
        if self.enable_frame_id_corruption:
            self.get_logger().info(f"  - Frame ID corruption: '{self.corrupt_frame_id}'")
        if self.enable_covariance_errors:
            self.get_logger().info(f"  - Covariance scaling: {self.covariance_scale}x")
        if self.enable_scale_errors:
            self.get_logger().info(f"  - Scale errors: Accel {self.scale_accel}, Gyro {self.scale_gyro}")
        self.get_logger().info("=" * 60)

    def _add_noise(self, value: float, std: float) -> float:
        """Add Gaussian noise to a value."""
        if std <= 0:
            return value
        return value + random.gauss(0, std)

    def _apply_axis_flip(self, value: float, flip: bool) -> float:
        """Flip sign if enabled."""
        return -value if flip else value

    def _corrupt_orientation(self, quat: Quaternion) -> Quaternion:
        """Apply various corruptions to orientation."""
        q = copy.deepcopy(quat)
        
        # Add some small orientation errors for realism
        if self.enable_axis_flip:
            # Rotate 90° around Y axis (swap X and Z with sign changes)
            if random.random() < 0.3:  # Occasional orientation corruption
                # Simple quaternion corruption: swap and flip components
                q.w, q.x, q.y, q.z = q.w, q.z, q.y, -q.x
                
        if self.enable_zero_bias:
            # Add small bias to quaternion (not physically accurate but demonstrates)
            # Normalize after corruption
            q.w += 0.01 * random.random()
            q.x += 0.01 * random.random()
            q.y += 0.01 * random.random()
            q.z += 0.01 * random.random()
            norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
            if norm > 0:
                q.w /= norm
                q.x /= norm
                q.y /= norm
                q.z /= norm
                
        return q

    def _imu_callback(self, msg: Imu) -> None:
        self.message_count += 1
        
        # Simulate message drops
        if self.enable_drop_messages:
            if random.random() < self.drop_rate:
                self.get_logger().debug(f"Dropping message #{self.message_count}")
                return

        # Create output message
        output = copy.deepcopy(msg)
        
        # Corrupt frame_id
        if self.enable_frame_id_corruption:
            output.header.frame_id = self.corrupt_frame_id
        
        # Add timestamp delay
        if self.enable_timestamp_delay:
            # Delay the timestamp by adding nanoseconds
            delay_ns = int(self.timestamp_delay_sec * 1e9)
            output.header.stamp.nanosec += delay_ns
            # Handle nanosecond overflow
            if output.header.stamp.nanosec >= 1e9:
                output.header.stamp.sec += int(output.header.stamp.nanosec // 1e9)
                output.header.stamp.nanosec = int(output.header.stamp.nanosec % 1e9)

        # Apply defects to linear acceleration
        if self.enable_scale_errors:
            msg.linear_acceleration.x *= self.scale_accel[0]
            msg.linear_acceleration.y *= self.scale_accel[1]
            msg.linear_acceleration.z *= self.scale_accel[2]
        
        if self.enable_zero_bias:
            output.linear_acceleration.x += self.bias_accel[0]
            output.linear_acceleration.y += self.bias_accel[1]
            output.linear_acceleration.z += self.bias_accel[2]
        
        if self.enable_axis_flip:
            output.linear_acceleration.x = self._apply_axis_flip(
                output.linear_acceleration.x, self.flip_accel_x)
            output.linear_acceleration.y = self._apply_axis_flip(
                output.linear_acceleration.y, self.flip_accel_y)
            output.linear_acceleration.z = self._apply_axis_flip(
                output.linear_acceleration.z, self.flip_accel_z)
        
        if self.enable_noise:
            output.linear_acceleration.x = self._add_noise(
                output.linear_acceleration.x, self.noise_accel_std)
            output.linear_acceleration.y = self._add_noise(
                output.linear_acceleration.y, self.noise_accel_std)
            output.linear_acceleration.z = self._add_noise(
                output.linear_acceleration.z, self.noise_accel_std)

        # Apply defects to angular velocity
        if self.enable_scale_errors:
            output.angular_velocity.x *= self.scale_gyro[0]
            output.angular_velocity.y *= self.scale_gyro[1]
            output.angular_velocity.z *= self.scale_gyro[2]
        
        if self.enable_zero_bias:
            output.angular_velocity.x += self.bias_gyro[0]
            output.angular_velocity.y += self.bias_gyro[1]
            output.angular_velocity.z += self.bias_gyro[2]
        
        if self.enable_axis_flip:
            output.angular_velocity.x = self._apply_axis_flip(
                output.angular_velocity.x, self.flip_gyro_x)
            output.angular_velocity.y = self._apply_axis_flip(
                output.angular_velocity.y, self.flip_gyro_y)
            output.angular_velocity.z = self._apply_axis_flip(
                output.angular_velocity.z, self.flip_gyro_z)
        
        if self.enable_noise:
            output.angular_velocity.x = self._add_noise(
                output.angular_velocity.x, self.noise_gyro_std)
            output.angular_velocity.y = self._add_noise(
                output.angular_velocity.y, self.noise_gyro_std)
            output.angular_velocity.z = self._add_noise(
                output.angular_velocity.z, self.noise_gyro_std)

        # Corrupt orientation
        if self.enable_zero_bias or self.enable_axis_flip:
            output.orientation = self._corrupt_orientation(output.orientation)

        # Corrupt covariance
        if self.enable_covariance_errors:
            # Scale covariance matrices
            for i in range(9):
                output.orientation_covariance[i] *= self.covariance_scale
                output.angular_velocity_covariance[i] *= self.covariance_scale
                output.linear_acceleration_covariance[i] *= self.covariance_scale
            
            # Make orientation covariance invalid (set first to -1) occasionally
            if self.message_count % 5 == 0:
                output.orientation_covariance[0] = -1.0

        # Publish the defective IMU data
        self.publisher.publish(output)
        
        # Log occasionally
        if self.message_count % 50 == 0:
            self.get_logger().debug(
                f"Processed {self.message_count} messages, "
                f"accel=({output.linear_acceleration.x:.3f}, {output.linear_acceleration.y:.3f}, {output.linear_acceleration.z:.3f})"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IMUDefectSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()