import mujoco
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Tuple, List, Optional
import logging
import tempfile
import os

# Assumes these files are in your project structure and importable
from dofbot.envs.dofbot_base_env import DofbotEnv
from dofbot.utils.scene_builder import SceneBuilder
from dofbot.utils.task_blueprints import TASK_BLUEPRINTS

class TableSceneEnv(DofbotEnv):
    """
    Enhanced dynamic environment for multi-object, multi-step robotic manipulation tasks.
    Improved state tracking, reward computation, and action handling.
    """
    def __init__(self, render_mode: str = None, **kwargs):
        # 1. SETUP LOGGING (moved to the beginning)
        self.logger = logging.getLogger(self.__class__.__name__)

        # 2. INITIALIZE CORE COMPONENTS
        self.scene_builder = SceneBuilder(
            base_scene_path="grasp_plan/dofbot/assets/scene/base_scene.xml",
            ycb_assets_path="grasp_plan/dofbot/assets/ycb"
        )
        self.task_blueprints = TASK_BLUEPRINTS

        # 3. EPISODE AND TASK TRACKING
        self.current_step = 0
        self.max_episode_steps = 500
        self.current_task_info = None
        self.task_sequence = []
        self.current_sequence_idx = 0
        
        # Enhanced grasping state
        self.grasped_object_name = None
        self.grasp_stability_counter = 0
        self.min_grasp_stability_steps = 5

        # 4. REWARD AND SUCCESS THRESHOLDS
        self.distance_threshold = 0.04
        self.lift_height_threshold = 0.08
        self.grasp_distance_threshold = 0.03
        self.place_distance_threshold = 0.05
        
        # 5. STATE TRACKING
        self.initial_object_positions = {}
        self.object_lifted_flags = {}
        self.sub_task_completion_flags = []
        
        # 6. OBSERVATION SPACE CONFIGURATION
        self.include_goal_info = True
        self.include_object_states = True

        # 7. INITIALIZE THE PARENT CLASS
        # Create a temporary file for the initial scene and pass its path
        self.model_file_path = self._create_dummy_scene()
        super().__init__(model_path=self.model_file_path, render_mode=render_mode, **kwargs)

        self.observation_space = self._get_extended_observation_space()
        
    @property
    def model_name(self) -> str:
        return self.model_file_path

    def _create_dummy_scene(self) -> str:
        """Create a minimal valid scene XML file for initialization and return its path."""
        xml_string = self._generate_dummy_xml()        

        # Save temp file in same directory as base_scene.xml so relative includes still work
        scene_dir = os.path.dirname(self.scene_builder.base_xml_path)
        os.makedirs(scene_dir, exist_ok=True)
        self.temp_file_path = os.path.join(scene_dir, "temp_scene.xml")

        with open(self.temp_file_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        
        abs_path = os.path.abspath(self.temp_file_path)  # make absolute
        self.logger.info(f"Created temporary dummy scene file: {abs_path}")
        return abs_path
        
    def _generate_dummy_xml(self) -> str:
        """Create a minimal valid scene for initialization."""
        try:
            # Generate the scene content, which likely has the <mujocoinclude> tag as its root
            return self.scene_builder.generate_scene_xml_for_task(self.task_blueprints['stack_blocks'])
            
        except Exception as e:
            self.logger.error(f"Failed to create dummy scene: {e}")
            # Return minimal XML as fallback
            return """<?xml version="1.0"?>
            <mujoco>
                <worldbody>
                    <body name="table" pos="0.3 0 0">
                        <geom name="table_geom" type="box" size="0.4 0.4 0.05" rgba="0.8 0.6 0.4 1"/>
                    </body>
                </worldbody>
            </mujoco>"""

    def reset_model(self) -> Dict[str, np.ndarray]:
        """Enhanced reset with better error handling and state initialization."""
        try:
            # 1. SELECT AND PREPARE A NEW TASK
            task_key = np.random.choice(list(self.task_blueprints.keys()))
            print(f"Selected task: {task_key}")

            # prob = np.array([0.25, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05])
            # template_id = np.random.choice(a=range(len(list(self.task_blueprints.keys()))), size=1, p=prob)[0]
            # task_key = list(self.task_blueprints.keys())[template_id]
            # print(f"Selected task: {task_key}")

            self.current_task_info = self.task_blueprints[task_key]
            
            # Validate task blueprint
            if not self.scene_builder.validate_task_blueprint(self.current_task_info):
                self.logger.warning(f"Invalid task blueprint for {task_key}, using fallback")
                task_key = 'stack_blocks'
                self.current_task_info = self.task_blueprints[task_key]
            
            self.logger.info(f"Starting new episode with task: {task_key}")
            
            # Parse temporal sequence
            self.task_sequence = self._parse_temporal_sequence(self.current_task_info)
            self.current_sequence_idx = 0
            self.sub_task_completion_flags = [False] * len(self.task_sequence)
            
            # Reset grasping state
            self.grasped_object_name = None
            self.grasp_stability_counter = 0
            self.object_lifted_flags = {}

            # 2. GENERATE AND LOAD THE NEW SCENE
            xml_string = self.scene_builder.generate_scene_xml_for_task(self.current_task_info)
            print("----- End of XML -----")
            
            # Create a new temporary file for this reset
            if hasattr(self, 'model_file_path') and os.path.exists(self.model_file_path):
                os.remove(self.model_file_path) # Clean up previous file

            with open(self.temp_file_path, 'w', encoding='utf-8') as f:
                f.write(xml_string)
            self.model_file_path = os.path.abspath(self.temp_file_path)
            
            try:
                self.model = mujoco.MjModel.from_xml_path(self.model_file_path)
                self.data = mujoco.MjData(self.model)
            except Exception as e:
                self.logger.error(f"Error loading model from XML for task: {task_key}")
                self.logger.error(f"XML content preview: {xml_string[:500]}...")
                raise e

            # 3. RE-ACQUIRE MODEL REFERENCES
            self._get_model_references()

            # Reset renderer if it exists
            if hasattr(self, 'mujoco_renderer') and self.mujoco_renderer is not None:
                self.mujoco_renderer.model = self.model
                self.mujoco_renderer.data = self.data

            # 4. RESET EPISODE STATE
            self.current_step = 0
            mujoco.mj_forward(self.model, self.data)

            # Store initial object positions for tracking
            self.initial_object_positions = {}
            for obj_name in self.object_body_ids:
                if obj_name in self.object_body_ids and self.object_body_ids[obj_name] != -1:
                    pos, _ = self._get_object_pose(obj_name)
                    self.initial_object_positions[obj_name] = pos.copy()
                    self.object_lifted_flags[obj_name] = False

            return self._get_obs()
            
        except Exception as e:
            self.logger.error(f"Error in reset_model: {e}")
            raise

    def _parse_temporal_sequence(self, task_info: dict) -> List[Dict]:
        """Parse temporal sequence into structured sub-tasks."""
        sequence = []
        temp_seq = task_info.get("temporal_sequence", [])
        
        for i, entry in enumerate(temp_seq):
            # First object: just place it (it should already be placed by scene builder)
            if i == 0:
                continue
            else:
                # Subsequent objects: grasp then place
                # sequence.extend([
                #     {"action": "grasp", "object": obj_name, "step": len(sequence)},
                #     {"action": "place", "object": obj_name, "target": "target_location", "step": len(sequence) + 1}
                # ])
                entry['step'] = len(sequence)
                sequence.append(entry)
        
        return sequence

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Enhanced step function with improved action handling and state updates."""
        # Process action
        gripper_action = 1.0 if action[-1] > 0.5 else 0.0
        arm_action = np.clip(action[:-1], -1, 1)  # Ensure arm actions are in valid range

        # Apply actions to actuators
        self.data.ctrl[:-1] = arm_action
        if hasattr(self.model, 'actuator') and len(self.data.ctrl) > len(arm_action):
            # Find gripper actuator and apply action
            try:
                gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'grip_actuator')
                if gripper_actuator_id != -1:
                    self.data.ctrl[-1] = gripper_action * self.model.actuator(gripper_actuator_id).gainprm[0]
                else:
                    self.data.ctrl[-1] = gripper_action
            except:
                self.data.ctrl[-1] = gripper_action

        # Step simulation
        self.do_simulation(self.data.ctrl, self.frame_skip)
        self.current_step += 1

        # Update grasping state
        self._update_grasping_state(gripper_action > 0.5)

        # Calculate results
        observation = self._get_obs()
        reward, info = self._evaluate_state()
        
        # Check termination conditions
        terminated = info.get("task_complete", False)
        truncated = self.current_step >= self.max_episode_steps

        # Add debug info
        info.update({
            "current_step": self.current_step,
            "current_subtask_idx": self.current_sequence_idx,
            "total_subtasks": len(self.task_sequence),
            "grasped_object": self.grasped_object_name,
            "grasp_stability": self.grasp_stability_counter
        })

        return observation, reward, terminated, truncated, info

    def _update_grasping_state(self, gripper_closed: bool):
        """Enhanced grasping state management with stability checking."""
        if not gripper_closed:
            # Gripper is open
            if self.grasped_object_name is not None:
                self.logger.debug(f"Released object: {self.grasped_object_name}")
            self.grasped_object_name = None
            self.grasp_stability_counter = 0
            return

        # Gripper is closed - check for grasp
        ee_pos, _ = self._get_end_effector_pose()
        
        # Find closest object within grasp range
        closest_object = None
        min_distance = float('inf')
        
        for obj_name in self.object_body_ids:
            if self.object_body_ids[obj_name] == -1:
                continue
                
            obj_pos, _ = self._get_object_pose(obj_name)
            distance = np.linalg.norm(ee_pos - obj_pos)
            
            if distance < self.grasp_distance_threshold and distance < min_distance:
                min_distance = distance
                closest_object = obj_name

        if closest_object is not None:
            if self.grasped_object_name == closest_object:
                self.grasp_stability_counter += 1
            else:
                self.grasped_object_name = closest_object
                self.grasp_stability_counter = 1
                self.logger.debug(f"Grasping object: {closest_object}")
        else:
            # No object in range
            if self.grasped_object_name is not None:
                # Check if we're still holding the object (it might have moved with gripper)
                if self.grasped_object_name in self.object_body_ids:
                    obj_pos, _ = self._get_object_pose(self.grasped_object_name)
                    if np.linalg.norm(ee_pos - obj_pos) > self.grasp_distance_threshold * 1.5:
                        self.grasped_object_name = None
                        self.grasp_stability_counter = 0

    def _get_model_references(self):
        """Enhanced model reference acquisition with error handling."""
        try:
            # Gripper and robot references
            self.end_effector_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'grip_site')
            if self.end_effector_site_id == -1:
                # Try alternative names
                for name in ['gripper_site', 'end_effector', 'ee_site']:
                    self.end_effector_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                    if self.end_effector_site_id != -1:
                        break
            
            # Gripper joint references
            self.gripper_joint_ids = []
            for joint_name in ['right_gripper_finger_joint', 'left_gripper_finger_joint', 'gripper_joint']:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id != -1:
                    self.gripper_joint_ids.append(joint_id)
            
            # Dynamic object references - need to map from unique names back to original names
            self.object_body_ids = {}
            self.target_site_ids = {}
            self.unique_to_original_name = {}  # Map unique names back to original names
            
            if self.current_task_info and 'objects' in self.current_task_info:
                # First, find all bodies that match object patterns
                for obj in self.current_task_info['objects']:
                    original_name = obj['name']
                    
                    # Look for bodies with this base name (handling unique suffixes)
                    found_body = False
                    for body_id in range(self.model.nbody):
                        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                        if body_name and (body_name == original_name or body_name.startswith(f"{original_name}_")):
                            # Use the actual body name found in the model
                            self.object_body_ids[original_name] = body_id
                            self.unique_to_original_name[body_name] = original_name
                            
                            # Look for corresponding target site
                            target_site_name = f"{body_name}_target"
                            target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, target_site_name)
                            self.target_site_ids[f"{original_name}_target"] = target_site_id
                            
                            found_body = True
                            break
                    
                    if not found_body:
                        self.logger.warning(f"Could not find body for object: {original_name}")
                        self.object_body_ids[original_name] = -1
                        self.target_site_ids[f"{original_name}_target"] = -1
                        
        except Exception as e:
            self.logger.error(f"Error getting model references: {e}")
            raise

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Enhanced observation construction with better error handling."""
        try:
            # Base observations from parent class
            obs = super()._get_obs()

            # End effector state
            if self.end_effector_site_id != -1:
                ee_pos, ee_quat = self._get_end_effector_pose()
                obs["end_effector_pos"] = ee_pos
                obs["end_effector_quat"] = ee_quat
            else:
                obs["end_effector_pos"] = np.zeros(3)
                obs["end_effector_quat"] = np.array([1, 0, 0, 0])

            # Object states
            if self.include_object_states:
                object_positions = []
                object_orientations = []
                
                for obj_name in self.object_body_ids:
                    if self.object_body_ids[obj_name] != -1:
                        pos, quat = self._get_object_pose(obj_name)
                        object_positions.extend(pos)
                        object_orientations.extend(quat)
                    else:
                        object_positions.extend([0, 0, 0])
                        object_orientations.extend([1, 0, 0, 0])
                
                obs["object_positions"] = np.array(object_positions)
                obs["object_orientations"] = np.array(object_orientations)

            # Gripper state
            obs["gripper_state"] = self._get_gripper_state()
            obs["is_grasping"] = float(self.grasped_object_name is not None)

            # Task and goal information
            if self.include_goal_info:
                obs["goal_info"] = self._get_goal_state()
                obs["task_progress"] = self._get_task_progress()
                
            # Sub-task information
            obs["current_subtask_idx"] = float(self.current_sequence_idx)
            obs["subtask_completion"] = np.array(self.sub_task_completion_flags, dtype=float)

            return obs
            
        except Exception as e:
            self.logger.error(f"Error constructing observation: {e}")
            # Return minimal observation to prevent crashes
            return {"error": True}

    def _get_gripper_state(self) -> float:
        """Get normalized gripper opening (0 = closed, 1 = open)."""
        if not self.gripper_joint_ids:
            return 0.0
        
        try:
            # Average position of gripper joints
            positions = [self.data.qpos[joint_id] for joint_id in self.gripper_joint_ids if joint_id != -1]
            if positions:
                return float(np.mean(positions))
            return 0.0
        except:
            return 0.0

    def _get_task_progress(self) -> np.ndarray:
        """Get overall task progress information."""
        total_subtasks = len(self.task_sequence)
        completed_subtasks = sum(self.sub_task_completion_flags)
        
        return np.array([
            float(completed_subtasks) / max(1, total_subtasks),  # Completion ratio
            float(self.current_sequence_idx) / max(1, total_subtasks),  # Current progress
            float(total_subtasks)  # Total subtasks
        ])

    def _get_goal_state(self) -> np.ndarray:
        """Enhanced goal state with better error handling."""
        if self.current_sequence_idx >= len(self.task_sequence):
            return np.zeros(3)  # No active goal

        try:
            current_sub_task = self.task_sequence[self.current_sequence_idx]
            target_obj_name = current_sub_task["object"]
            action_type = current_sub_task["action"]
            print(f"Current sub-task: {action_type} {target_obj_name}")
            
            if action_type == "grasp":
                # Goal is to reach the target object
                if target_obj_name in self.object_body_ids and self.object_body_ids[target_obj_name] != -1:
                    pos, _ = self._get_object_pose(target_obj_name)
                    return pos
                    
            elif action_type == "place":
                # Goal is to reach the placement target
                target_name = current_sub_task.get("target", "target_location")
                target_site_id = self.target_site_ids.get(f"{target_obj_name}_target")
                
                if target_site_id is not None and target_site_id != -1:
                    return self.data.site_xpos[target_site_id].copy()
                else:
                    # Fallback: use a default placement location
                    return np.array([0.3, 0.0, 0.1])

            return np.zeros(3)
            
        except Exception as e:
            self.logger.warning(f"Error getting goal state: {e}")
            return np.zeros(3)

    def _evaluate_state(self) -> Tuple[float, Dict]:
        """Enhanced reward calculation with detailed sub-task tracking."""
        if self.current_sequence_idx >= len(self.task_sequence):
            return 10.0, {"task_complete": True, "success": True}

        try:
            sub_task = self.task_sequence[self.current_sequence_idx]
            action_type = sub_task["action"]
            target_obj_name = sub_task["object"]

            reward = 0.0
            info = {
                "sub_task": f"{action_type} {target_obj_name}",
                "sub_task_success": False,
                "task_complete": False,
                "success": False
            }
            
            # Check if object exists in scene
            if target_obj_name not in self.object_body_ids or self.object_body_ids[target_obj_name] == -1:
                self.logger.warning(f"Target object {target_obj_name} not found in scene")
                return -1.0, info

            ee_pos, _ = self._get_end_effector_pose()
            obj_pos, _ = self._get_object_pose(target_obj_name)
            
            if action_type == "grasp":
                reward, info = self._evaluate_grasp_task(ee_pos, obj_pos, target_obj_name, info)
                
            elif action_type == "place":
                reward, info = self._evaluate_place_task(ee_pos, obj_pos, target_obj_name, info)

            # Check for task completion
            if info.get("sub_task_success", False):
                self.sub_task_completion_flags[self.current_sequence_idx] = True
                self.current_sequence_idx += 1
                
                if self.current_sequence_idx >= len(self.task_sequence):
                    info["task_complete"] = True
                    info["success"] = True
                    reward += 10.0  # Task completion bonus

            return reward, info
            
        except Exception as e:
            self.logger.error(f"Error evaluating state: {e}")
            return -1.0, {"error": True}

    def _evaluate_grasp_task(self, ee_pos: np.ndarray, obj_pos: np.ndarray, 
                           obj_name: str, info: Dict) -> Tuple[float, Dict]:
        """Evaluate grasping sub-task."""
        reward = 0.0
        
        # Phase 1: Reaching reward
        dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
        reach_reward = 1.0 - np.tanh(10.0 * dist_to_obj)
        reward += reach_reward
        
        # Phase 2: Grasping reward
        if (self.grasped_object_name == obj_name and 
            self.grasp_stability_counter >= self.min_grasp_stability_steps):
            
            reward += 2.0  # Stable grasp bonus
            
            # Phase 3: Lifting reward
            if obj_name in self.initial_object_positions:
                current_height = obj_pos[2]
                initial_height = self.initial_object_positions[obj_name][2]
                lift_height = current_height - initial_height
                
                if lift_height > self.lift_height_threshold:
                    if not self.object_lifted_flags.get(obj_name, False):
                        reward += 5.0  # First lift bonus
                        self.object_lifted_flags[obj_name] = True
                        info["sub_task_success"] = True
                    else:
                        reward += 1.0  # Continued lift reward
        
        return reward, info

    def _evaluate_place_task(self, ee_pos: np.ndarray, obj_pos: np.ndarray,
                           obj_name: str, info: Dict) -> Tuple[float, Dict]:
        """Evaluate placing sub-task."""
        reward = 0.0
        
        # Check if we're holding the correct object
        if self.grasped_object_name != obj_name:
            info["error"] = f"Not holding target object {obj_name}"
            return -0.5, info
        
        # Find target placement location
        target_site_id = self.target_site_ids.get(f"{obj_name}_target")
        if target_site_id is None or target_site_id == -1:
            # Use default placement logic
            target_pos = np.array([0.3, 0.0, 0.1])
        else:
            target_pos = self.data.site_xpos[target_site_id].copy()
        
        # Placement reward based on distance to target
        dist_to_target = np.linalg.norm(obj_pos - target_pos)
        place_reward = 2.0 - np.tanh(10.0 * dist_to_target)
        reward += place_reward
        
        # Success condition: object near target and gripper open
        gripper_open = self._get_gripper_state() < 0.3
        if dist_to_target < self.place_distance_threshold:
            if gripper_open:
                reward += 5.0  # Successful placement
                info["sub_task_success"] = True
                self.grasped_object_name = None  # Release object
            else:
                reward += 2.0  # Near target but haven't released
        
        return reward, info

    def _get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced end-effector pose with error handling."""
        if self.end_effector_site_id == -1:
            return np.zeros(3), np.array([1, 0, 0, 0])
        
        try:
            pos = self.data.site_xpos[self.end_effector_site_id].copy()
            # Get rotation matrix and convert to quaternion if needed
            rot_mat = self.data.site_xmat[self.end_effector_site_id].reshape(3, 3)
            # For simplicity, return flattened rotation matrix
            # In production, you might want to convert to quaternion
            return pos, rot_mat.flatten()
        except Exception as e:
            self.logger.warning(f"Error getting end-effector pose: {e}")
            return np.zeros(3), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

    def _get_object_pose(self, obj_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced object pose retrieval with error handling."""
        if obj_name not in self.object_body_ids:
            return np.zeros(3), np.array([1, 0, 0, 0])
        
        body_id = self.object_body_ids[obj_name]
        if body_id == -1:
            return np.zeros(3), np.array([1, 0, 0, 0])
        
        try:
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()
            return pos, quat
        except Exception as e:
            self.logger.warning(f"Error getting pose for object {obj_name}: {e}")
            return np.zeros(3), np.array([1, 0, 0, 0])

    def get_task_info(self) -> Dict:
        """Get information about the current task."""
        return {
            "task_name": getattr(self.current_task_info, 'name', 'unknown'),
            "description": self.current_task_info.get('description', ''),
            "total_subtasks": len(self.task_sequence),
            "current_subtask": self.current_sequence_idx,
            "completion_status": self.sub_task_completion_flags.copy(),
            "objects": list(self.object_body_ids.keys())
        }

    def render_debug_info(self):
        """Render debug information overlay (if using visual rendering)."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            # Add debug text to viewer
            debug_text = f"Task: {self.current_sequence_idx}/{len(self.task_sequence)}\n"
            debug_text += f"Grasped: {self.grasped_object_name}\n"
            debug_text += f"Stability: {self.grasp_stability_counter}\n"
            # Implementation depends on your viewer setup

    def _get_extended_observation_space(self) -> spaces.Dict:
        """
        Helper method to build the complete observation space.
        """
        base_space_dict = super().observation_space.spaces
        
        # Assuming a maximum number of objects for dynamic elements
        # You may need to adjust this or make it dynamic
        max_objects = 10 

        extended_space = spaces.Dict({
            **base_space_dict,
            "end_effector_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "end_effector_quat": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32), 
            "object_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(max_objects * 3,), dtype=np.float32),
            "object_orientations": spaces.Box(low=-np.inf, high=np.inf, shape=(max_objects * 4,), dtype=np.float32),
            "gripper_state": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "is_grasping": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "goal_info": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "task_progress": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "current_subtask_idx": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "subtask_completion": spaces.Box(low=0.0, high=1.0, shape=(max_objects * 2,), dtype=np.float32),
        })
        return extended_space