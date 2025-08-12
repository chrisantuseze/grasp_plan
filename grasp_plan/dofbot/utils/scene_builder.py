import xml.etree.ElementTree as ET
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class SceneBuilder:
    """
    Enhanced MuJoCo XML scene generator with better collision detection,
    proper object sizing, and improved placement logic.
    """
    def __init__(self, base_scene_path: str, ycb_assets_path: str):
        self.base_xml_path = base_scene_path
        self.ycb_assets_path = Path(ycb_assets_path)
        
        # Enhanced workspace bounds with safety margins
        self.workspace_bounds = {
            "center": ([0.2, 0.35], [-0.1, 0.1]),
            "left_side": ([0.15, 0.35], [0.1, 0.25]),
            "right_side": ([0.15, 0.35], [-0.25, -0.1]),
            "front": ([0.15, 0.25], [-0.15, 0.15]),
            "back": ([0.3, 0.4], [-0.15, 0.15]),
            "any": ([0.15, 0.4], [-0.25, 0.25])
        }
        self.table_z = 0.05
        self.min_object_spacing = 0.05  # Minimum distance between objects
        
        # Cache for object dimensions to avoid repeated XML parsing
        self.object_dimensions_cache = {}
        self._load_object_dimensions()

    def _load_object_dimensions(self):
        """Pre-load object dimensions from asset files for better placement."""
        dimensions_file = self.ycb_assets_path / "object_dimensions.json"
        if dimensions_file.exists():
            with open(dimensions_file, 'r') as f:
                self.object_dimensions_cache = json.load(f)
        else:
            # Default dimensions if file doesn't exist
            self.object_dimensions_cache = {
                "024_bowl": {"size": [0.08, 0.08, 0.04], "type": "container"},
                "031_spoon": {"size": [0.02, 0.15, 0.01], "type": "utensil"},
                "013_apple": {"size": [0.06, 0.06, 0.07], "type": "food"},
                "017_orange": {"size": [0.05, 0.06, 0.07], "type": "food"},
                "003_cracker_box": {"size": [0.08, 0.12, 0.18], "type": "box"},
                "006_mustard_bottle": {"size": [0.04, 0.04, 0.15], "type": "bottle"},
                "004_sugar_box": {"size": [0.06, 0.09, 0.12], "type": "box"},
                "blue_block": {"size": [0.05, 0.05, 0.05], "type": "block"},
                "red_block": {"size": [0.05, 0.05, 0.05], "type": "block"}
            }

    def _get_object_dimensions(self, object_name: str) -> Dict:
        """Get object dimensions, with fallback to default if not found."""
        return self.object_dimensions_cache.get(object_name, 
                                               {"size": [0.05, 0.05, 0.05], "type": "unknown"})

    def _check_collision(self, pos: np.ndarray, size: np.ndarray, 
                        occupied_positions: List[Dict]) -> bool:
        """Check if a position would collide with existing objects."""
        for occupied in occupied_positions:
            occupied_pos = occupied['pos']
            occupied_size = occupied['size']
            
            # Simple bounding box collision check
            min_distance = np.linalg.norm(size + occupied_size) / 2 + self.min_object_spacing
            actual_distance = np.linalg.norm(pos[:2] - occupied_pos[:2])  # Only check x,y
            
            if actual_distance < min_distance:
                return True
        return False

    def _get_placement_on_table(self, zone: str, occupied_positions: List[Dict], 
                               object_size: np.ndarray, max_attempts: int = 50) -> Optional[np.ndarray]:
        """Find collision-free placement on table with improved logic."""
        if zone not in self.workspace_bounds:
            zone = "any"
        
        x_range, y_range = self.workspace_bounds[zone]
        
        for _ in range(max_attempts):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            z = self.table_z + object_size[2] / 2  # Place on table surface
            
            pos = np.array([x, y, z])
            
            if not self._check_collision(pos, object_size, occupied_positions):
                return pos
        
        # Fallback: place in center with small random offset
        print(f"Warning: Could not find collision-free placement in zone {zone}, using fallback")
        center_x = sum(x_range) / 2 + random.uniform(-0.02, 0.02)
        center_y = sum(y_range) / 2 + random.uniform(-0.02, 0.02)
        return np.array([center_x, center_y, self.table_z + object_size[2] / 2])

    def _get_placement_near_object(self, target_pos: np.ndarray, target_size: np.ndarray,
                                  object_size: np.ndarray, offset: np.ndarray,
                                  occupied_positions: List[Dict]) -> np.ndarray:
        """Place object near another object with specified offset."""
        # Try the specified offset first
        pos = target_pos + offset
        pos[2] = self.table_z + object_size[2] / 2  # Ensure on table surface
        
        if not self._check_collision(pos, object_size, occupied_positions):
            return pos
        
        # If collision, try nearby positions
        for angle in np.linspace(0, 2*np.pi, 8):
            distance = np.linalg.norm(offset[:2]) if np.linalg.norm(offset[:2]) > 0 else 0.1
            offset_x = distance * np.cos(angle)
            offset_y = distance * np.sin(angle)
            pos = target_pos + np.array([offset_x, offset_y, 0])
            pos[2] = self.table_z + object_size[2] / 2
            
            if not self._check_collision(pos, object_size, occupied_positions):
                return pos
        
        # Fallback: place slightly away from target
        return target_pos + np.array([0.08, 0, object_size[2] / 2])

    def _get_placement_on_object(self, target_pos: np.ndarray, target_size: np.ndarray,
                               object_size: np.ndarray) -> np.ndarray:
        """Place object on top of another object."""
        z_offset = target_size[2] / 2 + object_size[2] / 2
        return target_pos + np.array([0, 0, z_offset])

    def _get_placement_in_container(self, target_pos: np.ndarray, target_size: np.ndarray,
                                  object_size: np.ndarray) -> np.ndarray:
        """Place object inside a container with realistic positioning."""
        # Ensure object fits in container
        max_x_offset = max(0, (target_size[0] - object_size[0]) / 2 - 0.01)
        max_y_offset = max(0, (target_size[1] - object_size[1]) / 2 - 0.01)
        
        dx = random.uniform(-max_x_offset, max_x_offset)
        dy = random.uniform(-max_y_offset, max_y_offset)
        dz = object_size[2] / 2 + 0.01  # Slightly above container bottom
        
        return target_pos + np.array([dx, dy, dz])
        

    def generate_scene_xml_for_task(self, task_blueprint: dict) -> str:
        """
        Enhanced scene generation with proper asset and body integration.
        """
        try:
            self.base_tree = ET.parse(self.base_xml_path)
            self.base_root = self.base_tree.getroot()
            self.worldbody = self.base_root.find('worldbody')
            self.asset_elem = self.base_root.find('asset')

            placed_objects = {}  # role -> {name, pos, size, type}
            occupied_table_positions = []
            object_counter = {}  # Track multiple instances of same object

            # Process placement rules in order
            for rule in task_blueprint['placement_rules']:
                role = rule['role']
                obj_info = next((obj for obj in task_blueprint['objects'] if obj['role'] == role), None)
                
                if not obj_info:
                    print(f"Warning: No object found for role '{role}'")
                    continue

                object_name = obj_info['name']
                obj_dims = self._get_object_dimensions(object_name)
                object_size = np.array(obj_dims['size'])
                
                # Handle multiple instances of the same object
                if object_name in object_counter:
                    object_counter[object_name] += 1
                else:
                    object_counter[object_name] = 0
                unique_name = f"{object_name}_{object_counter[object_name]}"
                
                position = self._place_object_by_rule(rule, object_size, placed_objects, occupied_table_positions)
                
                if position is not None:
                    # Generate slight random rotation for realism
                    rotation_z = random.uniform(-0.1, 0.1)  # Small rotation around Z
                    orientation = np.array([np.cos(rotation_z/2), 0, 0, np.sin(rotation_z/2)])
                    
                    self.add_object_to_xml(object_name, unique_name, position, orientation)
                    
                    # Add target site for this object (for placing tasks)
                    self.add_target_site(unique_name, position)
                    
                    placed_objects[role] = {
                        "name": unique_name,  # Use unique name for tracking
                        "original_name": object_name,
                        "pos": position, 
                        "size": object_size,
                        "type": obj_dims['type']
                    }
                    
                    # Track table positions for collision detection
                    if abs(position[2] - (self.table_z + object_size[2] / 2)) < 0.01:
                        occupied_table_positions.append({
                            'pos': position, 
                            'size': object_size
                        })

            return ET.tostring(self.base_root, encoding='unicode')
            
        except Exception as e:
            print(f"Error generating scene: {e}")
            raise

    def _place_object_by_rule(self, rule: dict, object_size: np.ndarray, 
                            placed_objects: dict, occupied_table_positions: List[dict]) -> Optional[np.ndarray]:
        """Place object according to the placement rule."""
        rule_type = rule['type']
        
        if rule_type == 'on_table':
            zone = rule.get('zone', 'any')
            return self._get_placement_on_table(zone, occupied_table_positions, object_size)
        
        elif rule_type == 'near_object':
            target_role = rule['target_role']
            if target_role not in placed_objects:
                raise ValueError(f"Target '{target_role}' for '{rule['role']}' must be placed first!")
            
            target_info = placed_objects[target_role]
            offset = np.array(rule.get('offset', [0.1, 0, 0]))
            return self._get_placement_near_object(
                target_info['pos'], target_info['size'], object_size, offset, occupied_table_positions
            )
        
        elif rule_type == 'on_object':
            target_role = rule['target_role']
            if target_role not in placed_objects:
                raise ValueError(f"Target '{target_role}' for '{rule['role']}' must be placed first!")
            
            target_info = placed_objects[target_role]
            return self._get_placement_on_object(target_info['pos'], target_info['size'], object_size)
        
        elif rule_type == 'in_container':
            target_role = rule['target_role']
            if target_role not in placed_objects:
                raise ValueError(f"Target '{target_role}' for '{rule['role']}' must be placed first!")
            
            target_info = placed_objects[target_role]
            if target_info['type'] != 'container':
                print(f"Warning: Placing object in non-container '{target_info['name']}'")
            
            return self._get_placement_in_container(target_info['pos'], target_info['size'], object_size)
        
        else:
            print(f"Unknown placement rule type: {rule_type}")
            return None

    def add_object_to_xml(self, object_name: str, unique_name: str, position: np.ndarray, orientation: np.ndarray):
        """
        Enhanced object addition with proper asset integration for MuJoCo object files.
        """
        object_xml_path = self.ycb_assets_path / object_name / "model.xml"
        
        if not object_xml_path.exists():
            raise FileNotFoundError(f"Asset file not found for object: {object_name} at {object_xml_path}")

        try:
            # Parse the object's complete MuJoCo model
            obj_tree = ET.parse(object_xml_path)
            obj_root = obj_tree.getroot()

            # Extract and modify assets
            self._add_asset(obj_root.find('asset'), unique_name, object_name)
            
            # Extract and modify the body
            self._add_worldbody(obj_root.find('worldbody'), unique_name, position, orientation, object_name)

        except ET.ParseError as e:
            print(f"Error parsing XML for {object_name}: {e}")
            raise
        except Exception as e:
            print(f"Error integrating object {object_name}: {e}")
            raise

    def _add_asset(self, obj_asset_elem: ET.Element, unique_name: str, object_name: str):
        """Helper to add asset elements with unique names and correct paths."""
        if obj_asset_elem is None:
            return 

        for child in obj_asset_elem:
            # Skip if this asset already exists
            existing_asset = None
            if child.get('name'):
                existing_asset = self.asset_elem.find(f".//{child.tag}[@name='{unique_name}_{child.tag}']")
            
            if existing_asset is not None:
                print(f"Asset {child.get('name')} already exists, skipping.")
                continue

            # Create unique names for assets to avoid conflicts
            if child.get('name'):
                child.set('name', f"{unique_name}_{child.tag}")
            
            # Update file paths to be relative to the object directory
            if child.get('file'):
                file_path = child.get('file')
                # If path doesn't start with the object name, prepend it

            # For mesh references, update the mesh name reference
            if child.tag == 'mesh' and child.get('name'):
                child.set('name', f"{unique_name}_mesh")
            
            # For material references  
            if child.tag == 'material' and child.get('name'):
                child.set('name', f"{unique_name}_material")
                # Update texture reference
                if child.get('texture'):
                    child.set('texture', f"{unique_name}_texture")
            
            # For texture
            if child.tag == 'texture' and child.get('name'):
                child.set('name', f"{unique_name}_texture")
            
            # Add the modified asset element to main scene
            self.asset_elem.append(child)

    def _add_worldbody(self, body_elem: ET.Element, unique_name: str, position: np.ndarray, orientation: np.ndarray, object_name: str):
        if body_elem is None:
            raise ValueError(f"No worldbody found in {object_name} model")
        
        obj_body = body_elem.find('body')
        if obj_body is None:
            raise ValueError(f"No body element found in {object_name} model")

        # Update body with unique naming and positioning
        obj_body.set('name', unique_name)
        obj_body.set('pos', ' '.join(f'{x:.6f}' for x in position))
        obj_body.set('quat', ' '.join(f'{x:.6f}' for x in orientation))
        
        # Update child elements with unique names
        for child in obj_body.iter():
            if child.get('name') and child != obj_body:
                if child.tag == 'site':
                    new_name = f"{unique_name}_center_site"
                else:
                    old_name = child.get('name')
                    if old_name.startswith(object_name):
                        last_part_of_old_name = old_name.split('_')[-1]
                        new_name = f"{unique_name}_{last_part_of_old_name}"
                    else:
                        new_name = f"{unique_name}_{old_name}"
                child.set('name', new_name)
            
            # Update mesh references in geoms
            if child.tag == 'geom' and child.get('mesh'):
                child.set('mesh', f"{unique_name}_mesh")
            
            # Update material references
            if child.get('material'):
                child.set('material', f"{unique_name}_material")

        # Add the modified body to the main scene
        self.worldbody.append(obj_body)

    def add_target_site(self, object_name: str, position: np.ndarray):
        """Add a target site for object placement tasks."""
        site = ET.SubElement(self.worldbody, 'site')
        site.set('name', f'{object_name}_target')
        site.set('pos', ' '.join(f'{x:.6f}' for x in position))
        site.set('size', '0.02 0.02 0.001')
        site.set('rgba', '0 1 0 0.3')  # Green, semi-transparent
        site.set('type', 'cylinder')

    def validate_task_blueprint(self, task_blueprint: dict) -> bool:
        """Validate task blueprint structure and dependencies."""
        required_keys = ['description', 'objects', 'placement_rules']
        if not all(key in task_blueprint for key in required_keys):
            return False
        
        # Check that all referenced roles exist
        object_roles = {obj['role'] for obj in task_blueprint['objects']}
        
        for rule in task_blueprint['placement_rules']:
            if rule['role'] not in object_roles:
                print(f"Invalid rule: role '{rule['role']}' not found in objects")
                return False
            
            if 'target_role' in rule and rule['target_role'] not in object_roles:
                print(f"Invalid rule: target_role '{rule['target_role']}' not found in objects")
                return False
        
        return True