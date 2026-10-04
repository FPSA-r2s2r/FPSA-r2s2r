import json
import cv2
import numpy as np
from pathlib import Path

class EpisodeWriter:
    def __init__(self, episode_dir, 
                 if_agent_view  = True,
                 if_eye_in_hand = False,
                 fps = 20, extra_meta = None):
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self._if_agent_view  = if_agent_view
        self._if_eye_in_hand = if_eye_in_hand
        self.fps = fps
        self.agentview_writer = None
        self.eyeinhand_writer = None

        self.extra_meta = extra_meta or {}
        self.lowdim = {
            "robot0_eef_pos": [],
            "robot0_eef_quat": [],
            "robot0_gripper_qpos": [],
            "action": [],
            "timestamp": [],
        }

    def _make_writer(self, path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    def add_step(self, obs, action, timestamp):

        if self._if_agent_view: 
            agent = obs["agentview_image"]
            if self.agentview_writer is None:
                h, w = agent.shape[:2]
                self.agentview_writer = self._make_writer(
                    self.episode_dir / "agentview.mp4", w, h, self.fps
                )
            # cv2.VideoWriter expects BGR input.
            self.agentview_writer.write(agent[..., ::-1])
            
        if self._if_eye_in_hand:
            eye = obs["robot0_eye_in_hand_image"]
            if self.eyeinhand_writer is None:
                h, w = eye.shape[:2]
                self.eyeinhand_writer = self._make_writer(
                    self.episode_dir / "eye_in_hand.mp4", w, h, self.fps
                )
            
            self.eyeinhand_writer.write(eye[..., ::-1])
        # record lowdim state and action
        self.lowdim["robot0_eef_pos"].append(obs["robot0_eef_pos"])
        self.lowdim["robot0_eef_quat"].append(obs["robot0_eef_quat"])
        self.lowdim["robot0_gripper_qpos"].append(obs["robot0_gripper_qpos"])
        self.lowdim["action"].append(action)
        self.lowdim["timestamp"].append(float(timestamp))

    def close(self, success=True):
        if self.agentview_writer is not None:
            self.agentview_writer.release()
        if self.eyeinhand_writer is not None:
            self.eyeinhand_writer.release()

        np.savez_compressed(
            self.episode_dir / "lowdim.npz",
            robot0_eef_pos=np.asarray(self.lowdim["robot0_eef_pos"], dtype=np.float32),
            robot0_eef_quat=np.asarray(self.lowdim["robot0_eef_quat"], dtype=np.float32),
            robot0_gripper_qpos=np.asarray(self.lowdim["robot0_gripper_qpos"], dtype=np.float32),
            action=np.asarray(self.lowdim["action"], dtype=np.float32),
            timestamp=np.asarray(self.lowdim["timestamp"], dtype=np.float64),
        )

        meta = {
            "success": bool(success),
            "num_steps": int(len(self.lowdim["timestamp"])),
            "fps": int(self.fps),
            **self.extra_meta, 
        }
        with open(self.episode_dir / "episode_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
