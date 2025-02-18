from typing import List, Any

import torch
from easydict import EasyDict
from .motion_lib import MotionLib, fix_motion_fps, LoadedMotions
import numpy as np
from typing import Literal


class G1_MotionLib(MotionLib):
    def __init__(
        self,
        motion_file,
        dof_body_ids,
        dof_offsets,
        key_body_ids,
        device="cpu",
        ref_height_adjust: float = 0,
        target_frame_rate: int = 30,
        w_last: bool = True,
        create_text_embeddings: bool = False,
        spawned_scene_ids: List[str] = None,
        fix_motion_heights: bool = True,
        skeleton_tree: Any = None,
    ):

        super().__init__(
            motion_file=motion_file,
            dof_body_ids=dof_body_ids,
            dof_offsets=dof_offsets,
            key_body_ids=key_body_ids,
            device=device,
            ref_height_adjust=ref_height_adjust,
            target_frame_rate=target_frame_rate,
            w_last=w_last,
            create_text_embeddings=create_text_embeddings,
            spawned_scene_ids=spawned_scene_ids,
            fix_motion_heights=fix_motion_heights,
            skeleton_tree=skeleton_tree,
        )

        motions = self.state.motions
        self.register_buffer(
            "dof_pos",
            torch.cat([m.dof_pos for m in motions], dim=0).to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

    @staticmethod
    def _load_motion_file(motion_file):
        motion = EasyDict(torch.load(motion_file, weights_only=False))
        return motion

    def _compute_motion_dof_vels(self, motion):
        # We pre-compute the dof vels in h1_humanoid_batch fk.
        return motion.dof_vels

    def fix_motion_heights(self, motion, skeleton_tree):
        body_heights = motion.global_translation[..., 2].clone()
        # TODO: this is a bit hacky and hardcoded for the current defined key body ids
        # left ankle: self.key_body_ids[0]
        # right ankle: self.key_body_ids[1]
        # body_heights[:, self.key_body_ids[0]] -= 0.06
        # body_heights[:, self.key_body_ids[1]] -= 0.06
        min_height = body_heights.min()

        motion.global_translation[..., 2] -= min_height
        # motion.global_translation[..., 2] += 0.01
        return motion

    def _load_motions(self, motion_file, target_frame_rate):
        if self.create_text_embeddings:
            from transformers import AutoTokenizer, XCLIPTextModel

            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        motions = []
        motion_lengths = []
        motion_dt = []
        motion_num_frames = []
        text_embeddings = []
        has_text_embeddings = []
        (
            motion_files,
            motion_weights,
            motion_timings,
            motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
            supported_scene_ids,
        ) = self._fetch_motion_files(motion_file)

        num_motion_files = len(motion_files)

        for f in range(num_motion_files):
            curr_file = motion_files[f]

            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )
            curr_motion = self._load_motion_file(curr_file)

            curr_motion = fix_motion_fps(
                curr_motion,
                motion_fpses[f],
                target_frame_rate,
                self.skeleton_tree,
            )
            motion_fpses[f] = float(curr_motion.fps)

            if self.fix_heights:
                curr_motion = self.fix_motion_heights(curr_motion, self.skeleton_tree)

            curr_motion.dof_pos[:, 4] = - curr_motion.dof_pos[:, 0] - curr_motion.dof_pos[:, 3]
            curr_motion.dof_pos[:, 10] = - curr_motion.dof_pos[:, 6] - curr_motion.dof_pos[:, 9]

            curr_dt = 1.0 / motion_fpses[f]

            num_frames = curr_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fpses[f] * (num_frames - 1)

            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            joint_reorder_indices = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
            curr_motion.dof_pos = curr_motion.dof_pos[:, joint_reorder_indices]
            curr_motion.dof_vels = curr_motion.dof_vels[:, joint_reorder_indices]

            body_reorder_indices = [0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 5, 11, 17, 24, 6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29]
            curr_motion.global_angular_velocity = curr_motion.global_angular_velocity[:, body_reorder_indices, :]
            curr_motion.global_rotation = curr_motion.global_rotation[:, body_reorder_indices, :]
            curr_motion.global_rotation_mat = curr_motion.global_rotation_mat[:, body_reorder_indices, :, :]  # ?
            curr_motion.global_velocity = curr_motion.global_velocity[:, body_reorder_indices, :]
            curr_motion.global_translation = curr_motion.global_translation[:, body_reorder_indices, :]
            curr_motion.local_rotation = curr_motion.local_rotation[:, body_reorder_indices, :]

            curr_motion.global_rotation = convert_quat(curr_motion.global_rotation, to="wxyz")
            curr_motion.local_rotation = convert_quat(curr_motion.local_rotation, to="wxyz")

            motions.append(curr_motion)
            motion_lengths.append(curr_len)

        num_sub_motions = len(sub_motion_to_motion)

        for f in range(num_sub_motions):
            # Incase start/end weren't provided, set to (0, motion_length)
            motion_f = sub_motion_to_motion[f]
            if motion_timings[f][1] == -1:
                motion_timings[f][1] = motion_lengths[motion_f]

            motion_timings[f][1] = min(
                motion_timings[f][1], motion_lengths[motion_f]
            )  # CT hack: fix small timing differences

            assert (
                motion_timings[f][0] < motion_timings[f][1]
            ), f"Motion start {motion_timings[f][0]} >= motion end {motion_timings[f][1]} in motion {motion_f}"

            if self.create_text_embeddings and motion_labels[f][0] != "":
                with torch.inference_mode():
                    inputs = tokenizer(
                        motion_labels[f],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    outputs = model(**inputs)
                    pooled_output = outputs.pooler_output  # pooled (EOS token) states
                    text_embeddings.append(pooled_output)  # should be [3, 512]
                    has_text_embeddings.append(True)
            else:
                text_embeddings.append(
                    torch.zeros((3, 512), dtype=torch.float32)
                )  # just hold something temporary
                has_text_embeddings.append(False)

        motion_lengths = torch.tensor(
            motion_lengths, device=self._device, dtype=torch.float32
        )

        motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self._device
        )
        motion_weights /= motion_weights.sum()

        motion_timings = torch.tensor(
            motion_timings, dtype=torch.float32, device=self._device
        )

        sub_motion_to_motion = torch.tensor(
            sub_motion_to_motion, dtype=torch.long, device=self._device
        )

        ref_respawn_offsets = torch.tensor(
            ref_respawn_offsets, dtype=torch.float32, device=self._device
        )

        motion_fpses = torch.tensor(
            motion_fpses, device=self._device, dtype=torch.float32
        )
        motion_dt = torch.tensor(motion_dt, device=self._device, dtype=torch.float32)
        motion_num_frames = torch.tensor(motion_num_frames, device=self._device)

        text_embeddings = torch.stack(text_embeddings).detach().to(device=self._device)
        has_text_embeddings = torch.tensor(
            has_text_embeddings, dtype=torch.bool, device=self._device
        )

        self.state = LoadedMotions(
            motions=tuple(motions),
            motion_lengths=motion_lengths,
            motion_weights=motion_weights,
            motion_timings=motion_timings,
            motion_fps=motion_fpses,
            motion_dt=motion_dt,
            motion_num_frames=motion_num_frames,
            motion_files=tuple(motion_files),
            sub_motion_to_motion=sub_motion_to_motion,
            ref_respawn_offsets=ref_respawn_offsets,
            text_embeddings=text_embeddings,
            has_text_embeddings=has_text_embeddings,
            supported_scene_ids=supported_scene_ids,
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        num_sub_motions = self.num_sub_motions()
        total_trainable_len = self.get_total_trainable_length()

        print(
            "Loaded {:d} sub motions with a total trainable length of {:.3f}s.".format(
                num_sub_motions, total_trainable_len
            )
        )


def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)
    # check if input is numpy array (we support this backend since some classes use numpy)
    if isinstance(quat, np.ndarray):
        # use numpy functions
        if to == "xyzw":
            # wxyz -> xyzw
            return np.roll(quat, -1, axis=-1)
        else:
            # xyzw -> wxyz
            return np.roll(quat, 1, axis=-1)
    else:
        # convert to torch (sanity check)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=float)
        # convert to specified quaternion type
        if to == "xyzw":
            # wxyz -> xyzw
            return quat.roll(-1, dims=-1)
        else:
            # xyzw -> wxyz
            return quat.roll(1, dims=-1)
