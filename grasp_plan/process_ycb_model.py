import os
from pathlib import Path

# This is an XML template for a generic, movable object in MuJoCo.
YCB_OBJECT_XML_TEMPLATE = """
<mujoco model="{object_name}">
  <asset>
    <texture name="{object_name}_texture" type="2d" file="{texture_file_path}" />
    <material name="{object_name}_material" texture="{object_name}_texture" specular="0.5" shininess="0.3"/>
    <mesh name="{object_name}_mesh" file="{mesh_file_path}" scale="0.5 0.5 0.5" />
  </asset>

  <worldbody>
    <body name="{object_name}" pos="0 0 0" quat="1 0 0 0">
      <joint name="{object_name}_joint" type="free" damping="0.01" armature="0.001"/>
      
      <geom name="{object_name}_geom"
            type="mesh"
            mesh="{object_name}_mesh"
            mass="0.2" 
            contype="1" 
            conaffinity="1" 
            solimp="0.95 0.99 0.001" 
            solref="0.004 1"
            friction="0.5 0.1 0.1"/>
            
      <site name="{object_name}_center_site" type="sphere" size="0.005" rgba="1 0 0 0.5"/>
    </body>

  </worldbody>
</mujoco>
"""

def create_ycb_model_xml(ycb_root_dir: str, object_name: str):
    """
    Generates a model.xml file for a YCB object from its mesh file.

    Args:
        ycb_root_dir: The root directory where YCB objects are stored (e.g., 'path/to/ycb').
        object_name: The name of the object folder (e.g., '003_cracker_box').
    """
    ycb_root = Path(ycb_root_dir)
    object_path = ycb_root / object_name
    
    # We assume the main mesh is in a 'google16k' subfolder and is an OBJ.
    # Adjust this path to match your directory structure.
    mesh_path = object_path / "google_16k" / "textured.obj"
    
    if not mesh_path.exists():
        print(f"Warning: Mesh file not found for {object_name} at {mesh_path}. Skipping.")
        return

    # Generate the XML content from the template
    xml_content = YCB_OBJECT_XML_TEMPLATE.format(
        object_name=object_name,
        texture_file_path=str(f"../ycb/{object_name}/google_16k/texture_map.png"),
        # mesh_file_path=str(mesh_path.relative_to(object_path)) # Path relative to the XML
        mesh_file_path=str(f"../ycb/{object_name}/google_16k/textured.obj")
    )

    # Save the generated XML to model.xml
    output_xml_path = object_path / "model.xml"
    with open(output_xml_path, 'w') as f:
        f.write(xml_content)
    print(f"Successfully created model.xml for {object_name}")

# --- Example Usage ---
# You would run this once to pre-process all your YCB objects.
if __name__ == '__main__':
    YCB_ASSETS_DIRECTORY = "grasp_plan/dofbot/assets/ycb"
    
    # Get all object folders in the directory
    object_folders = [d for d in os.listdir(YCB_ASSETS_DIRECTORY) if os.path.isdir(os.path.join(YCB_ASSETS_DIRECTORY, d))]
    
    for folder in object_folders:
        create_ycb_model_xml(YCB_ASSETS_DIRECTORY, folder)