import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.joint2motor_idx = config["joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(
                config["default_angles"], dtype=np.float32)

            if "torso_idx" in config:
                self.torso_idx = config["torso_idx"]

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.history_length = config["history_length"]
            self.cmd_range = config["cmd_range"]

            if "motion_file" in config:
                self.motion_file = config["motion_file"]
            else:
                self.motion_file = None

            if "use_height_map" in config:
                self.use_height_map = config["use_height_map"]
                self.height_map_size = config["height_map_size"]
                self.flat_height = config["flat_height"]
            else:
                self.motion_file = False
