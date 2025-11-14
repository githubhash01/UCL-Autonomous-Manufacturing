import mujoco
import mujoco.viewer

# Path to your URDF file
URDF_PATH = "/home/hashim/Desktop/Simulation/Genesis/models/x500/model.urdf"

# Load the URDF model
model = mujoco.MjModel.from_xml_path(URDF_PATH)
data = mujoco.MjData(model)

# Launch the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("✅ Loaded model successfully. Use mouse to rotate, scroll to zoom.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
