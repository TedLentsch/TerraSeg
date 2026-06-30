import time

import rclpy
import torch
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data
from terraseg import TerraSegPredictor

from .pointcloud_conversion import (
    pointcloud2_to_xyz_tensor,
    xyz_and_labels_to_pointcloud2,
)


class TerraSegNode(Node):
    """
    ROS2 node that runs TerraSeg ground segmentation on an incoming LiDAR stream.

    Subscribes to:
        - ``<input_topic>`` (``sensor_msgs/PointCloud2``) : Incoming LiDAR scans in the
          TerraSeg-standardized frame (z = 0 approximately ground-aligned, +x forward).

    Publishes:
        - ``<output_topic>`` (``sensor_msgs/PointCloud2``) : The same XYZ points plus an
          added uint8 ``label`` field (0 = ground, 1 = non-ground).

    Parameters (declared with sensible defaults; override via launch YAML):
        - ``variant`` (str) : ``"B"`` or ``"S"``. Default ``"S"``.
        - ``checkpoint_path`` (str) : Either a local filesystem path to a trained
          ``best.pth``, or a Hugging Face URI of the form
          ``hf://<user>/<repo>/<filename>`` (e.g.
          ``hf://TedLentsch/TerraSeg/terraseg_s.pth``). Required.
        - ``hf_revision`` (str) : Optional revision (branch, tag, or commit hash) when
          loading from Hugging Face. Default ``""`` (latest on ``main``).
        - ``decision_threshold`` (float) : Sigmoid threshold. ``-1.0`` defers to the value
          stored in the checkpoint (recommended).
        - ``grid_size`` (float) : PTv3 voxel grid size. Unit: meters. Default ``0.05``.
        - ``input_topic`` (str) : Input PointCloud2 topic. Default ``/lidar/points``.
        - ``output_topic`` (str) : Output labeled PointCloud2 topic. Default
          ``/terraseg/segmented``.
        - ``device`` (str) : Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
          Default ``"cuda:0"``.
        - ``compile_model`` (bool) : Wrap the model with ``torch.compile`` for an extra
          inference speedup. Adds a one-time compilation cost at startup. Default ``True``.

    TerraSeg runs in FP32. Lower-precision dtypes (BF16, FP16) are not supported because
    PTv3's sparse-convolution path becomes numerically unstable at reduced precision.
    """

    def __init__(self):
        super().__init__("terraseg_node")

        # Declare parameters.
        self.declare_parameter("variant", "S")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("hf_revision", "")
        self.declare_parameter("decision_threshold", -1.0)
        self.declare_parameter("grid_size", 0.05)
        self.declare_parameter("input_topic", "/lidar/points")
        self.declare_parameter("output_topic", "/terraseg/segmented")
        # Frame the cloud is transformed into before inference. TerraSeg expects the
        # cloud roughly ground-aligned (z = 0 at ground, +x forward), which a roof or
        # otherwise-offset sensor frame is not. Set to "" to disable and feed the cloud
        # in its native frame unchanged.
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("compile_model", False)
        self.declare_parameter("tf32", True)

        variant = str(self.get_parameter("variant").value)
        checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        hf_revision = str(self.get_parameter("hf_revision").value) or None
        decision_threshold = float(self.get_parameter("decision_threshold").value)
        self.grid_size = float(self.get_parameter("grid_size").value)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        device_str = str(self.get_parameter("device").value)
        self.target_frame = str(self.get_parameter("target_frame").value) or None
        compile_model = bool(self.get_parameter("compile_model").value)
        tf32 = bool(self.get_parameter("tf32").value)

        if not checkpoint_path:
            raise RuntimeError(
                "Parameter 'checkpoint_path' is required (local file path or 'hf://...' URI). "
                "Set it in the launch YAML."
            )

        # Build predictor.
        self.get_logger().info(
            f"Loading TerraSeg-{variant} from '{checkpoint_path}' onto '{device_str}' "
            f"(compile={compile_model}, tf32={tf32})..."
        )
        self.predictor = TerraSegPredictor(
            variant=variant,
            checkpoint_path=checkpoint_path,
            device=device_str,
            decision_thres=(decision_threshold if decision_threshold >= 0.0 else None),
            compile_model=compile_model,
            tf32=tf32,
            hf_revision=hf_revision,
        )
        self.get_logger().info(
            f"Predictor ready. decision threshold={self.predictor.decision_thres:.3f}."
        )

        # TF: cache the static transform from the cloud frame into target_frame.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._tf_cache: dict[str, torch.Tensor] = {}

        # Subscriber + publisher.
        self.sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.on_pointcloud,
            qos_profile=qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(PointCloud2, output_topic, qos_profile=10)
        self.get_logger().info(
            f"Subscribed to '{input_topic}', publishing on '{output_topic}'."
        )


    def _lookup_transform_matrix(self, source_frame: str) -> torch.Tensor | None:
        """
        Return a cached (4, 4) homogeneous transform mapping points from ``source_frame``
        into ``self.target_frame`` as a float32 tensor on the predictor device, or None
        if the transform is unavailable.
        """
        if source_frame in self._tf_cache:
            return self._tf_cache[source_frame]
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        # Quaternion (x, y, z, w) to rotation matrix.
        x, y, z, w = q.x, q.y, q.z, q.w
        R = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=torch.float32,
            device=self.predictor.device,
        )
        M = torch.eye(4, dtype=torch.float32, device=self.predictor.device)
        M[:3, :3] = R
        M[:3, 3] = torch.tensor([t.x, t.y, t.z], dtype=torch.float32, device=self.predictor.device)
        self._tf_cache[source_frame] = M
        self.get_logger().info(
            f"Cached transform '{source_frame}' -> '{self.target_frame}'."
        )
        return M

    def on_pointcloud(
        self,
        msg: PointCloud2,
    ) -> None:
        """
        Per-scan callback: parse PointCloud2, run TerraSeg, publish labeled PointCloud2.

        Args:
            msg (sensor_msgs.msg.PointCloud2) : Incoming LiDAR scan.
        """
        t_start = time.monotonic()

        device = self.predictor.device
        coord = pointcloud2_to_xyz_tensor(msg=msg, device=device)
        if coord.shape[0] == 0:
            self.get_logger().warn("Received empty PointCloud2; skipping.")
            return

        # Transform into the ground-aligned target frame so the height / forward-axis
        # features match what TerraSeg expects. Skipped if target_frame is unset or the
        # incoming cloud is already in it.
        out_header = msg.header
        if self.target_frame and msg.header.frame_id != self.target_frame:
            M = self._lookup_transform_matrix(msg.header.frame_id)
            if M is None:
                self.get_logger().warn(
                    f"No transform '{msg.header.frame_id}' -> '{self.target_frame}' yet; "
                    "skipping scan.",
                    throttle_duration_sec=2.0,
                )
                return
            ones = torch.ones((coord.shape[0], 1), dtype=torch.float32, device=device)
            coord_h = torch.cat([coord, ones], dim=1)
            coord = (coord_h @ M.T)[:, :3].contiguous()
            out_header.frame_id = self.target_frame

        pred_labels = self.predictor.predict(coord=coord, grid_size=self.grid_size)

        out_msg = xyz_and_labels_to_pointcloud2(
            coord=coord, pred_labels=pred_labels, header=out_header
        )
        self.pub.publish(out_msg)

        dt_ms = 1000.0 * (time.monotonic() - t_start)
        self.get_logger().debug(
            f"Scan handled: {coord.shape[0]} points, {dt_ms:.1f} ms end-to-end."
        )


def main(args=None) -> None:
    """
    ROS2 entry point.

    Args:
        args (list or None) : Forwarded to :func:`rclpy.init`. Default: ``None``.
    """
    rclpy.init(args=args)
    node = TerraSegNode()
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
