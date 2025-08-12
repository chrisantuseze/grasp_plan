"""
Enhanced task blueprints with improved placement rules, 
better object relationships, and more realistic task sequences.
"""

TASK_BLUEPRINTS = {
    "set_the_table": {
        "description": "Set up a dining table with bowl, fork, and spoon arranged properly.",
        "difficulty": "medium",
        "objects": [
            {"name": "024_bowl", "role": "container_1", "type": "container"},
            {"name": "030_fork", "role": "utensil_1", "type": "utensil"},
            {"name": "031_spoon", "role": "utensil_2", "type": "utensil"}
        ],
        "placement_rules": [
            # Bowl placed first in center
            {"role": "container_1", "type": "on_table", "zone": "center"},
            # Fork placed to the left of bowl
            {"role": "utensil_1", "type": "near_object", "target_role": "container_1", 
             "offset": [-0.08, 0.0, 0.0], "description": "place fork to left of bowl"},
            # Spoon placed to the right of bowl  
            {"role": "utensil_2", "type": "near_object", "target_role": "container_1", 
             "offset": [0.08, 0.0, 0.0], "description": "place spoon to right of bowl"}
        ],
        "temporal_sequence": [
            {"object": "024_bowl", "action": "place", "target": "center_table"},
            {"object": "030_fork", "action": "grasp"},
            {"object": "030_fork", "action": "place", "target": "left_of_bowl"},
            {"object": "031_spoon", "action": "grasp"},
            {"object": "031_spoon", "action": "place", "target": "right_of_bowl"}
        ],
        "success_criteria": {
            "bowl_placed": True,
            "utensils_arranged": True,
            "proper_spacing": 0.06
        }
    },

    "prepare_a_snack": {
        "description": "Prepare a snack by placing food items into a bowl.",
        "difficulty": "medium",
        "objects": [
            {"name": "024_bowl", "role": "container_1", "type": "container"},
            {"name": "013_apple", "role": "food_1", "type": "food"},
            {"name": "003_cracker_box", "role": "food_2", "type": "food"}
        ],
        "placement_rules": [
            # Bowl placed first
            {"role": "container_1", "type": "on_table", "zone": "center"},
            # Food items start scattered, will be moved into bowl
            {"role": "food_1", "type": "on_table", "zone": "left_side"},
            {"role": "food_2", "type": "on_table", "zone": "right_side"}
        ],
        "temporal_sequence": [
            {"object": "024_bowl", "action": "place", "target": "center_table"},
            {"object": "013_apple", "action": "grasp"},
            {"object": "013_apple", "action": "place", "target": "in_bowl"},
            {"object": "003_cracker_box", "action": "grasp"},
            {"object": "003_cracker_box", "action": "place", "target": "in_bowl"}
        ],
        "success_criteria": {
            "items_in_container": ["013_apple", "003_cracker_box"],
            "container_stable": True
        }
    },

    "stack_blocks": {
        "description": "Create a stable stack by placing the red block on top of the blue block.",
        "difficulty": "easy",
        "objects": [
            {"name": "blue_block", "role": "base_block", "type": "block"},
            {"name": "red_block", "role": "mid_block", "type": "block"},
            {"name": "green_block", "role": "top_block", "type": "block"}
        ],
        "placement_rules": [
            # Base block placed first
            {"role": "base_block", "type": "on_table", "zone": "center"},
            # Top block starts elsewhere, will be stacked
            {"role": "mid_block", "type": "on_table", "zone": "left_side"},
            {"role": "top_block", "type": "on_table", "zone": "right_side"}
        ],
        "temporal_sequence": [
            {"object": "blue_block", "action": "place", "target": "center_table"},
            {"object": "red_block", "action": "grasp"},
            {"object": "red_block", "action": "place", "target": "on_blue_block"},
            {"object": "green_block", "action": "grasp"},
            {"object": "green_block", "action": "place", "target": "on_red_block"}
        ],
        "success_criteria": {
            "stacked_correctly": True,
            "stack_stable": True,
            "alignment_tolerance": 0.02
        }
    },

    "sort_items": {
        "description": "Sort items by placing specific objects into designated containers.",
        "difficulty": "hard",
        "objects": [
            {"name": "024_bowl", "role": "container_1", "type": "container"},
            {"name": "024_bowl", "role": "container_2", "type": "container"},  # Note: same model, different instance
            {"name": "006_mustard_bottle", "role": "item_1", "type": "bottle"},
            {"name": "004_sugar_box", "role": "item_2", "type": "box"}
        ],
        "placement_rules": [
            # Two containers in different zones
            {"role": "container_1", "type": "on_table", "zone": "left_side"},
            {"role": "container_2", "type": "on_table", "zone": "right_side"},
            # Items start in center, need to be sorted
            {"role": "item_1", "type": "on_table", "zone": "center"},
            {"role": "item_2", "type": "on_table", "zone": "center"}
        ],
        "temporal_sequence": [
            {"object": "024_bowl", "action": "place", "target": "left_table", "instance": 1},
            {"object": "024_bowl", "action": "place", "target": "right_table", "instance": 2},
            {"object": "006_mustard_bottle", "action": "grasp"},
            {"object": "006_mustard_bottle", "action": "place", "target": "in_left_container"},
            {"object": "004_sugar_box", "action": "grasp"},
            {"object": "004_sugar_box", "action": "place", "target": "in_right_container"}
        ],
        "success_criteria": {
            "correct_sorting": {
                "006_mustard_bottle": "container_1",
                "004_sugar_box": "container_2"
            },
            "containers_stable": True
        }
    },

    "pack_lunch": {
        "description": "Pack a lunch by placing multiple items into a lunchbox container.",
        "difficulty": "medium",
        "objects": [
            {"name": "024_bowl", "role": "lunchbox", "type": "container"},
            {"name": "013_apple", "role": "fruit", "type": "food"},
            {"name": "006_mustard_bottle", "role": "condiment", "type": "bottle"}
        ],
        "placement_rules": [
            # Lunchbox (bowl) placed first
            {"role": "lunchbox", "type": "on_table", "zone": "center"},
            # Items start scattered around table
            {"role": "fruit", "type": "on_table", "zone": "front"},
            {"role": "condiment", "type": "on_table", "zone": "back"}
        ],
        "temporal_sequence": [
            {"object": "024_bowl", "action": "place", "target": "center_table"},
            {"object": "013_apple", "action": "grasp"},
            {"object": "013_apple", "action": "place", "target": "in_lunchbox"},
            {"object": "006_mustard_bottle", "action": "grasp"},
            {"object": "006_mustard_bottle", "action": "place", "target": "in_lunchbox"}
        ],
        "success_criteria": {
            "all_items_packed": True,
            "container_not_overfilled": True,
            "stable_packing": True
        }
    },

    "clean_workspace": {
        "description": "Clean up the workspace by organizing scattered items into containers.",
        "difficulty": "hard",
        "objects": [
            {"name": "024_bowl", "role": "cleanup_container", "type": "container"},
            {"name": "017_orange", "role": "scattered_1", "type": "utensil"},
            {"name": "031_spoon", "role": "scattered_2", "type": "utensil"},
            {"name": "013_apple", "role": "scattered_3", "type": "food"}
        ],
        "placement_rules": [
            # Container placed strategically
            {"role": "cleanup_container", "type": "on_table", "zone": "center"},
            # Items scattered randomly
            {"role": "scattered_1", "type": "on_table", "zone": "left_side"},
            {"role": "scattered_2", "type": "on_table", "zone": "right_side"},
            {"role": "scattered_3", "type": "on_table", "zone": "front"}
        ],
        "temporal_sequence": [
            {"object": "024_bowl", "action": "place", "target": "center_table"},
            {"object": "017_orange", "action": "grasp"},
            {"object": "017_orange", "action": "place", "target": "in_container"},
            {"object": "031_spoon", "action": "grasp"},
            {"object": "031_spoon", "action": "place", "target": "in_container"},
            {"object": "013_apple", "action": "grasp"},
            {"object": "013_apple", "action": "place", "target": "in_container"}
        ],
        "success_criteria": {
            "all_items_collected": True,
            "workspace_clear": True,
            "organized_storage": True
        }
    },

    "build_tower": {
        "description": "Build a tower by stacking multiple objects in order of size.",
        "difficulty": "expert",
        "objects": [
            {"name": "004_sugar_box", "role": "tower_base", "type": "box"},  # Largest
            {"name": "blue_block", "role": "tower_middle", "type": "block"},  # Medium
            {"name": "013_apple", "role": "tower_top", "type": "food"}  # Smallest/lightest
        ],
        "placement_rules": [
            # All items start scattered
            {"role": "tower_base", "type": "on_table", "zone": "center"},
            {"role": "tower_middle", "type": "on_table", "zone": "left_side"},
            {"role": "tower_top", "type": "on_table", "zone": "right_side"}
        ],
        "temporal_sequence": [
            {"object": "004_sugar_box", "action": "place", "target": "center_table"},
            {"object": "blue_block", "action": "grasp"},
            {"object": "blue_block", "action": "place", "target": "on_sugar_box"},
            {"object": "013_apple", "action": "grasp"},
            {"object": "013_apple", "action": "place", "target": "on_blue_block"}
        ],
        "success_criteria": {
            "correct_stacking_order": ["004_sugar_box", "blue_block", "013_apple"],
            "tower_stable": True,
            "height_achieved": 0.15,  # Minimum tower height
            "alignment_tolerance": 0.03
        }
    }
}

# Task difficulty and learning progression
TASK_PROGRESSION = {
    "beginner": ["stack_blocks"],
    "intermediate": ["prepare_a_snack", "pack_lunch", "set_the_table"],
    "advanced": ["sort_items", "clean_workspace"],
    "expert": ["build_tower"]
}

# Task categories for curriculum learning
TASK_CATEGORIES = {
    "stacking": ["stack_blocks", "build_tower"],
    "containment": ["prepare_a_snack", "pack_lunch", "clean_workspace"],
    "arrangement": ["set_the_table", "sort_items"],
    "multi_step": ["sort_items", "clean_workspace", "build_tower"]
}

def get_tasks_by_difficulty(difficulty: str) -> list:
    """Get list of tasks by difficulty level."""
    return TASK_PROGRESSION.get(difficulty, [])

def get_tasks_by_category(category: str) -> list:
    """Get list of tasks by category."""
    return TASK_CATEGORIES.get(category, [])

def validate_all_blueprints() -> dict:
    """Validate all task blueprints and return validation results."""
    results = {}
    for task_name, blueprint in TASK_BLUEPRINTS.items():
        results[task_name] = validate_blueprint(blueprint)
    return results

def validate_blueprint(blueprint: dict) -> dict:
    """Validate a single task blueprint."""
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ['description', 'objects', 'placement_rules', 'temporal_sequence']
    for field in required_fields:
        if field not in blueprint:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    # Check object roles consistency
    object_roles = {obj['role'] for obj in blueprint['objects']}
    
    # Check placement rules reference valid roles
    for rule in blueprint['placement_rules']:
        if rule['role'] not in object_roles:
            errors.append(f"Placement rule references unknown role: {rule['role']}")
        if 'target_role' in rule and rule['target_role'] not in object_roles:
            errors.append(f"Placement rule references unknown target_role: {rule['target_role']}")
    
    # Check temporal sequence references valid objects
    for step in blueprint['temporal_sequence']:
        if isinstance(step, dict) and 'object' in step:
            obj_name = step['object']
            if not any(obj['name'] == obj_name for obj in blueprint['objects']):
                errors.append(f"Temporal sequence references unknown object: {obj_name}")
    
    # Check for dependency order in placement rules
    dependent_roles = set()
    for rule in blueprint['placement_rules']:
        if 'target_role' in rule:
            dependent_roles.add(rule['role'])
    
    # Warnings for potential issues
    if len(blueprint['objects']) > 5:
        warnings.append("Task has many objects (>5), may be complex for learning")
    
    if len(blueprint['temporal_sequence']) > 10:
        warnings.append("Long temporal sequence (>10 steps), may be difficult")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "num_objects": len(blueprint['objects']),
            "num_placement_rules": len(blueprint['placement_rules']),
            "sequence_length": len(blueprint['temporal_sequence']),
            "difficulty": blueprint.get('difficulty', 'unknown')
        }
    }