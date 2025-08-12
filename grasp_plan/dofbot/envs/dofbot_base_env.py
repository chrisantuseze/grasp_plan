import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv as MjEnvGym # Alias to avoid confusion
from gymnasium.utils import seeding
import mujoco # For direct MuJoCo API calls if needed
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing_extensions import TypeAlias
from typing import Any, Callable, Literal, Optional, SupportsFloat

# Optional: Add a cached_property if you want to delay observation space definition
from functools import cached_property
from typing import Any, SupportsFloat

RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"

class DofbotEnv(MjEnvGym):
    """
    A custom MuJoCo environment for the DOFBOT-pro robot,
    following the standard of gymnasium.envs.mujoco.MujocoEnv.
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"], # Depth array needs specific camera setup in XML and renderer
        "render_fps": 80, #100, # Target rendering framerate
    }

    @cached_property
    def observation_space(self) -> spaces.Dict:
        qpos_size = qvel_size = 6  # Assuming DOFBOT has 6 DOF, adjust if necessary

        return spaces.Dict({
            "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(qpos_size,), dtype=np.float32),
            "qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(qvel_size,), dtype=np.float32),
            "wrist_camera": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "front_camera": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
        })
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        frame_skip: int = 5, # How many MuJoCo steps per environment step
        render_mode: Optional[str] = None,
        camera_name: Optional[str] = None, # If you want to use a specific camera in XML for rendering
        camera_id: Optional[int] = None, # Or by ID
    ) -> None:
        # Call the base class constructor
        # It will load the model, data, and set up the viewer if render_mode='human'

        super().__init__(
            model_path=model_path, # Pass the XML path here
            frame_skip=frame_skip,
            observation_space=self.observation_space, # Pass the defined observation space
            render_mode=render_mode,
            width=640,
            height=480,
            camera_name=camera_name,
            camera_id=camera_id,
            default_camera_config={"trackbodyid": 0}
        )

        # Action space: Directly control the robot's actuators
        # Your dofbot.xml uses position actuators, so action maps to actuator control
        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0],
            high=self.model.actuator_ctrlrange[:, 1],
            dtype=np.float32
        )

        # Store initial robot state for resetting
        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)

        # Inside __init__ or reset_model
        self.arm_joint_names = [
            "Arm1_Joint", "Arm2_Joint", "Arm3_Joint",
            "Arm4_Joint", "Arm5_Joint", "grip_joint"
        ]

        # Get joint indices in qpos and qvel
        self.arm_qpos_ids = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.arm_joint_names
        ])
        self.arm_qvel_ids = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.arm_joint_names
        ])


    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def seed(self, seed: int) -> list[int]:
        """Seeds the environment.

        Args:
            seed: The seed to use.

        Returns:
            The seed used inside a 1 element list.
        """
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        # assert self.goal_space
        # self.goal_space.seed(seed)
        return [seed]
    
    def get_camera_info(self, camera_name):
        """
        Get information about the front camera
        """
        camera_id = mujoco.mj_name2id(
            self.model, 
            mujoco.mjtObj.mjOBJ_CAMERA, 
            camera_name
        )
        
        if camera_id == -1:
            return "Camera 'front_camera' not found"
        
        # Get camera position and orientation
        cam_pos = self.data.cam_xpos[camera_id].copy()
        cam_mat = self.data.cam_xmat[camera_id].copy().reshape(3, 3)
        
        return {
            'camera_id': camera_id,
            'position': cam_pos,
            'rotation_matrix': cam_mat,
            'fovy': self.model.cam_fovy[camera_id]
        }
    
    def get_camera_image(self, camera_name: str = None, width: int = 640, height: int = 480):
        """
        Render the environment from a specific camera using direct MuJoCo rendering.
        
        Args:
            camera_name: Name of the camera to render from. If None, uses the environment's current camera.
            width: Image width
            height: Image height
        
        Returns:
            RGB image array from the specified camera
        """
        if camera_name is None:
            # Use the environment's current renderer
            return self.mujoco_renderer.render(render_mode='rgb_array')
        
        # Get camera ID from name
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in the model")
        
        # Create renderer for this camera if it doesn't exist
        renderer_key = f"camera_renderer_{width}_{height}"
        if not hasattr(self, '_camera_renderers'):
            self._camera_renderers = {}
        
        if renderer_key not in self._camera_renderers:
            self._camera_renderers[renderer_key] = mujoco.Renderer(self.model, height=height, width=width)
        
        renderer = self._camera_renderers[renderer_key]
        
        # Update renderer with current simulation state and specific camera
        renderer.update_scene(self.data, camera=camera_id)
        
        # Render and return image
        image = renderer.render()
        return image

    def get_wrist_and_front_images(self, width: int = 640, height: int = 480):
        """
        Get images from both wrist_camera and front_camera.
        
        Args:
            width: Image width
            height: Image height
        
        Returns:
            Dictionary with 'wrist_camera' and 'front_camera' keys containing RGB images
        """
        images = {}
        
        try:
            images['wrist_camera'] = self.get_camera_image('wrist_camera', width, height)
        except ValueError as e:
            print(f"Camera Warning: {e}")
            images['wrist_camera'] = None
        
        try:
            images['front_camera'] = self.get_camera_image('front_camera', width, height)
        except ValueError as e:
            print(f"Camera Warning: {e}")
            images['front_camera'] = None
        
        return images

    def list_available_cameras(self):
        """
        List all available cameras in the model.
        
        Returns:
            List of camera names
        """
        camera_names = []
        for i in range(self.model.ncam):
            camera_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if camera_name:
                camera_names.append(camera_name)
        return camera_names

    def close(self):
        """Override close method to cleanup renderers"""
        # Clean up custom renderers
        if hasattr(self, '_camera_renderers'):
            for renderer in self._camera_renderers.values():
                renderer.close()
            self._camera_renderers.clear()
        
        # Call parent close method
        super().close()

    def _get_obs(self) -> npt.NDArray[np.float64]:
        """
        Gathers the current observation from the MuJoCo simulation.
        This is similar to Meta-World's _get_curr_obs_combined_no_goal but simpler.
        """
        qpos = self.data.qpos[self.arm_qpos_ids].astype(np.float32)
        qvel = self.data.qvel[self.arm_qvel_ids].astype(np.float32)

        obs = {
            "qpos": qpos,
            "qvel": qvel,
            # "wrist_camera": np.zeros((3, 480, 640)).astype(np.uint8),
            # "front_camera": np.zeros((3, 480, 640)).astype(np.uint8),
            "wrist_camera": self.get_camera_image("wrist_camera").astype(np.uint8),
            "front_camera": self.get_camera_image("front_camera").astype(np.uint8),
        }
        return obs

    def reset_model(self) -> npt.NDArray[np.float64]:
        """
        Resets the MuJoCo simulation to its initial state.
        This method is called by the base Gymnasium's reset().
        """
        # Set the simulation state to the initial recorded qpos and qvel
        self.set_state(self.init_qpos, self.init_qvel)

        # If you need to randomize initial joint positions, do it here
        # E.g., self.data.qpos[joint_id] = self.np_random.uniform(low, high)
    
    # def reset_model(self):
    #     self.data.qpos[:6] = np.array([0.8, 0.3, 0.2, -0.1, 0.4, 0])  # or whatever your DOF is
    #     self.data.qvel[:] = 0
    #     self.set_state(self.data.qpos, self.data.qvel)
    #     return self._get_obs()

    
    def reset(
        self, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: The seed to use. Ignored, use seed() instead.
            options: Additional options to pass to the environment. Ignored.

        Returns:
            The (obs, info) tuple.
        """
        self.reset_model()
        obs, info = super().reset(seed=seed, options=options)

        if obs is None:
            obs = {}

        if info is None:
            info = {}

        observation = self._get_obs()  # Get the observation from the current state
        obs.update(observation)

        return obs, info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Applies the action, steps the simulation, and returns the next state.
        """
        # Ensure action is within bounds and apply to actuators
        assert len(action) == 6, f"Actions should be size 6, got {len(action)}"

        # (The base class MujocoEnv's do_simulation takes care of clipping)
        self.data.ctrl[:] = action

        # Advance the simulation by self.frame_skip steps
        # This method is provided by gymnasium.envs.mujoco.MujocoEnv
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # Get observation, reward, and done flags
        observation = self._get_obs()

        reward, info = self.evaluate_state(observation, action)

        terminated = False # Robot-only env likely doesn't terminate easily
        # truncated = False # Or truncate by episode length (e.g., self.curr_path_length counter)
        truncated = info.get("truncated", False)  # Use info dict to determine truncation

        # The rendering for 'human' mode is handled by the base class's render() method
        # which is called by the gymnasium wrapper if render_mode is set.
        # You don't need self.viewer.sync() directly here if you rely on the base render().

        return (observation, reward, terminated, truncated, info)
    
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        """Does the heavy-lifting for step() -- namely, calculating reward and populating the info dict with training metrics.

        Returns:
            Tuple of reward between 0 and 10 and a dictionary which contains useful metrics (success,
                near_object, grasp_success, grasp_reward, in_place_reward,
                obj_to_target, unscaled_reward)
        """
        raise NotImplementedError

    # The render() and close() methods are implicitly handled by MujocoEnv
    # You can override them if you need custom rendering logic (e.g., saving video)
    # but for basic 'human' viewing, the base class handles it.