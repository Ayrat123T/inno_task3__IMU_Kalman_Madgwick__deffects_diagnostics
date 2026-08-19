#!/usr/bin/env python3
"""Runnable 1D Kalman with dual input/output.

Subscribes: 
  - /ouster/imu (reference/clean)
  - /ouster/imu_defective (defective)
Publishes:
  - /imu/kalman_1d_clean (filtered clean)
  - /imu/kalman_1d_defective (filtered defective)

Runs Kalman filter on both clean and defective IMU data.
"""

from __future__ import annotations

import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

SUPPORTED_SIGNALS = {
    "linear_acceleration.x",
    "linear_acceleration.y",
    "linear_acceleration.z",
    "angular_velocity.x",
    "angular_velocity.y",
    "angular_velocity.z",
}


class Kalman1DNode(Node):
    def __init__(self) -> None:
        super().__init__("kalman_1d_node")

        # Parameters
        self.declare_parameter("clean_input_topic", "/ouster/imu")
        self.declare_parameter("defective_input_topic", "/ouster/imu_defective")
        self.declare_parameter("clean_output_topic", "/imu/kalman_1d_clean")
        self.declare_parameter("defective_output_topic", "/imu/kalman_1d_defective")
        self.declare_parameter("signal", "linear_acceleration.x")
        self.declare_parameter("process_variance", 0.01)
        self.declare_parameter("measurement_variance", 0.64)
        self.declare_parameter("initial_estimate", 0.0)
        self.declare_parameter("initial_error_variance", 1.0)

        self.clean_input_topic = str(self.get_parameter("clean_input_topic").value)
        self.defective_input_topic = str(self.get_parameter("defective_input_topic").value)
        self.clean_output_topic = str(self.get_parameter("clean_output_topic").value)
        self.defective_output_topic = str(self.get_parameter("defective_output_topic").value)
        self.signal = str(self.get_parameter("signal").value)

        if self.signal not in SUPPORTED_SIGNALS:
            raise ValueError(
                f"Unsupported signal {self.signal}; choose one of {sorted(SUPPORTED_SIGNALS)}"
            )

        self.process_variance = float(self.get_parameter("process_variance").value)
        self.measurement_variance = float(self.get_parameter("measurement_variance").value)
        
        # Two separate filter states for clean and defective data
        self.clean_state = {
            'estimate': float(self.get_parameter("initial_estimate").value),
            'variance': float(self.get_parameter("initial_error_variance").value)
        }
        self.defective_state = {
            'estimate': float(self.get_parameter("initial_estimate").value),
            'variance': float(self.get_parameter("initial_error_variance").value)
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
        self.get_logger().info("1D Kalman Filter with Dual Input/Output")
        self.get_logger().info(f"Signal: {self.signal}")
        self.get_logger().info(f"Q: {self.process_variance:.3f}, R: {self.measurement_variance:.3f}")
        self.get_logger().info(f"Clean input: {self.clean_input_topic}")
        self.get_logger().info(f"Defective input: {self.defective_input_topic}")
        self.get_logger().info(f"Clean output: {self.clean_output_topic}")
        self.get_logger().info(f"Defective output: {self.defective_output_topic}")
        self.get_logger().info("=" * 60)

    def _get_measurement(self, msg: Imu) -> float:
        group_name, axis = self.signal.split(".")
        return float(getattr(getattr(msg, group_name), axis))

    def _set_measurement(self, msg: Imu, value: float) -> None:
        group_name, axis = self.signal.split(".")
        setattr(getattr(msg, group_name), axis, float(value))

    def _filter_measurement(self, measurement: float, state: dict) -> float:
        """Apply Kalman filter to a measurement using provided state."""
        # Prediction
        x_pred = state['estimate']
        P_pred = state['variance'] + self.process_variance

        # Kalman gain
        K = P_pred / (P_pred + self.measurement_variance)

        # Correction
        innovation = measurement - x_pred
        state['estimate'] = x_pred + K * innovation

        # Covariance correction
        state['variance'] = (1.0 - K) * P_pred

        return state['estimate']

    def _clean_callback(self, msg: Imu) -> None:
        """Process clean IMU data."""
        measurement = self._get_measurement(msg)
        filtered = self._filter_measurement(measurement, self.clean_state)

        output = copy.deepcopy(msg)
        self._set_measurement(output, filtered)
        self.clean_publisher.publish(output)

    def _defective_callback(self, msg: Imu) -> None:
        """Process defective IMU data."""
        measurement = self._get_measurement(msg)
        filtered = self._filter_measurement(measurement, self.defective_state)

        output = copy.deepcopy(msg)
        self._set_measurement(output, filtered)
        self.defective_publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Kalman1DNode()
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