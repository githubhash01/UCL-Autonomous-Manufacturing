import genesis as gs

# Initialize Genesis
gs.init(backend=gs.gpu)   # or gs.cpu if you don't have a GPU

# Simulation scene
scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(dt=0.01)
)

# Path to your URDF model
URDF_PATH = "/home/hashim/Desktop/Simulation/Genesis/models/x500/model.urdf"

# Load the URDF as an asset
urdf_asset = gs.asset.URDF(
    file=URDF_PATH,
    fixed_base=True,       # Keep it from falling
    merge_fixed_links=True # Optional: merge static parts
)

# Add the model to the scene
robot = scene.add_entity(urdf_asset, pose=gs.Transform(position=[0, 0, 0.1]))

# Start the simulation
while scene.viewer.is_running():
    scene.step()
