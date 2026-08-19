#!/usr/bin/env python3
"""Runnable Madgwick with dual input/output.

Subscribes:
  - /ouster/imu (reference/clean)
  - /ouster/imu_defective (defective)
Publishes:
  - /imu/madgwick_clean
  - /imu/madgwick_defective

Runs Madgwick filter on both clean and defective IMU data.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class MadgwickNode(Node):
    def __init__(self) -> None:
        super().__init__("madgwick_node")

        # Parameters
        self.declare_parameter("clean_input_topic", "/ouster/imu")
        self.declare_parameter("defective_input_topic", "/ouster/imu_defective")
        self.declare_parameter("clean_output_topic", "/imu/madgwick_clean")
        self.declare_parameter("defective_output_topic", "/imu/madgwick_defective")
        self.declare_parameter("beta", 0.1)

        self.clean_input_topic = str(self.get_parameter("clean_input_topic").value)
        self.defective_input_topic = str(self.get_parameter("defective_input_topic").value)
        self.clean_output_topic = str(self.get_parameter("clean_output_topic").value)
        self.defective_output_topic = str(self.get_parameter("defective_output_topic").value)
        self.beta = float(self.get_parameter("beta").value)

        # Two separate filter states for clean and defective data
        self.clean_state = {
            'q': [1.0, 0.0, 0.0, 0.0],  # [w, x, y, z]
            'previous_time': None
        }
        self.defective_state = {
            'q': [1.0, 0.0, 0.0, 0.0],
            'previous_time': None
        }

        # Publishers
        self.clean_publisher = self.create_publisher(Imu, self.clean_output_topic, 10)
        self.defective_publisher = self.create_publisher(Imu, self.defective_output_topic, 10)

        # Subscribers
        self.clean_subscription = self.create_subscription(
            Imu, self.clean_input_topic, self._clean_callback, 10
        )
        self.defective_subscription = self.create_subscription(
            Imu, self.defective_input_topic, self._defective_callback, 10
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("Madgwick Filter with Dual Input/Output")
        self.get_logger().info(f"Beta: {self.beta:.3f}")
        self.get_logger().info(f"Clean input: {self.clean_input_topic}")
        self.get_logger().info(f"Defective input: {self.defective_input_topic}")
        self.get_logger().info(f"Clean output: {self.clean_output_topic}")
        self.get_logger().info(f"Defective output: {self.defective_output_topic}")
        self.get_logger().info("=" * 60)

    @staticmethod
    def _stamp_to_seconds(msg: Imu) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _compute_orientation(self, msg: Imu, state: dict):
        """Compute orientation using Madgwick filter for given state."""
        current_time = self._stamp_to_seconds(msg)

        # Compute dt
        if state['previous_time'] is None:
            dt = 0.0
        else:
            dt = current_time - state['previous_time']
        state['previous_time'] = current_time

        # If dt is zero or negative, return current state
        if dt <= 0.0:
            return tuple(state['q'])

        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        gx = float(msg.angular_velocity.x)
        gy = float(msg.angular_velocity.y)
        gz = float(msg.angular_velocity.z)

        # Normalize accelerometer vector
        acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if acc_norm == 0.0:
            return tuple(state['q'])
        ax /= acc_norm
        ay /= acc_norm
        az /= acc_norm

        qw, qx, qy, qz = state['q']

        # Quaternion derivative from gyroscope
        q_dot_gyro_w = 0.5 * (-qx * gx - qy * gy - qz * gz)
        q_dot_gyro_x = 0.5 * (qw * gx + qy * gz - qz * gy)
        q_dot_gyro_y = 0.5 * (qw * gy - qx * gz + qz * gx)
        q_dot_gyro_z = 0.5 * (qw * gz + qx * gy - qy * gx)

        # Madgwick gradient correction
        f1 = 2.0 * (qx * qz - qw * qy) - ax
        f2 = 2.0 * (qw * qx + qy * qz) - ay
        f3 = 2.0 * (0.5 - qy * qy - qz * qz) - az

        J11 = -2.0 * qy
        J12 = 2.0 * qz
        J13 = -2.0 * qw
        J14 = 2.0 * qx
        J21 = 2.0 * qx
        J22 = 2.0 * qw
        J23 = 2.0 * qz
        J24 = 2.0 * qy
        J31 = 0.0
        J32 = -4.0 * qx
        J33 = -4.0 * qy
        J34 = 0.0

        grad_w = J11 * f1 + J21 * f2 + J31 * f3
        grad_x = J12 * f1 + J22 * f2 + J32 * f3
        grad_y = J13 * f1 + J23 * f2 + J33 * f3
        grad_z = J14 * f1 + J24 * f2 + J34 * f3

        grad_norm = math.sqrt(grad_w * grad_w + grad_x * grad_x +
                              grad_y * grad_y + grad_z * grad_z)
        if grad_norm > 0.0:
            grad_w /= grad_norm
            grad_x /= grad_norm
            grad_y /= grad_norm
            grad_z /= grad_norm

        q_dot_w = q_dot_gyro_w - self.beta * grad_w
        q_dot_x = q_dot_gyro_x - self.beta * grad_x
        q_dot_y = q_dot_gyro_y - self.beta * grad_y
        q_dot_z = q_dot_gyro_z - self.beta * grad_z

        # Integrate
        qw_new = qw + q_dot_w * dt
        qx_new = qx + q_dot_x * dt
        qy_new = qy + q_dot_y * dt
        qz_new = qz + q_dot_z * dt

        norm = math.sqrt(qw_new * qw_new + qx_new * qx_new +
                         qy_new * qy_new + qz_new * qz_new)
        if norm > 0.0:
            state['q'] = [qw_new / norm, qx_new / norm, qy_new / norm, qz_new / norm]

        return tuple(state['q'])

    def _publish_orientation(self, msg: Imu, publisher, state) -> None:
        """Publish orientation for a given state."""
        quaternion = self._compute_orientation(msg, state)
        output = copy.deepcopy(msg)

        if quaternion is None:
            output.orientation_covariance[0] = -1.0
        else:
            qw, qx, qy, qz = quaternion
            output.orientation.w = float(qw)
            output.orientation.x = float(qx)
            output.orientation.y = float(qy)
            output.orientation.z = float(qz)
            output.orientation_covariance = [
                2.5e-3, 0.0, 0.0,
                0.0, 2.5e-3, 0.0,
                0.0, 0.0, 5.0e-3,
            ]

        publisher.publish(output)

    def _clean_callback(self, msg: Imu) -> None:
        """Process clean IMU data."""
        self._publish_orientation(msg, self.clean_publisher, self.clean_state)

    def _defective_callback(self, msg: Imu) -> None:
        """Process defective IMU data."""
        self._publish_orientation(msg, self.defective_publisher, self.defective_state)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MadgwickNode()
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