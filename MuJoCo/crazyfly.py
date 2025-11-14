import numpy as np
import mujoco
import time
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

XML_PATH = '/home/hashim/Desktop/Simulation/MuJoCo/bitcraze_crazyflie_2/scene.xml'

class PIDController:
    def __init__(self, dt, kp, ki, kd, setpoint):
        self.dt = dt 
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def compute(self, measured_value):
        error = self.setpoint - measured_value
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output
    

# Simple Drone class to load model and data
class Drone:
  
    def __init__(self, xml_path, hover_bias=0.26487):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.hover_bias = hover_bias
        self.dt = self.model.opt.timestep

        # IMU Sensor Indices
        self._gyro_i  = int(self.model.sensor('body_gyro').adr[0])
        self._accel_i = int(self.model.sensor('body_linacc').adr[0])
        self._quat_i  = int(self.model.sensor('body_quat').adr[0])

        # PID Thrust Controller Gains
        self.thrust_KP = 0.8 
        self.thrust_KI = 0.05
        self.thrust_KD = 0.3

        # PID Torque Controller Gains 
        self.torque_KP = 0.05
        self.torque_KI = 0.0
        self.torque_KD = 0.02

        # Thrust PID Controller for altitude 
        self.altitude_pid = PIDController(
            dt=self.dt,
            kp=self.thrust_KP, 
            ki=self.thrust_KI, 
            kd=self.thrust_KD, 
            setpoint=0.0
        )

    
    """
    Low-Level Drone Interface
    """
    # Actuation Methods
    def set_control(self, cntrl_values:np.ndarray) -> None:
        """
        Set control inputs for the drone.
        Args:
            cntrl_values (numpy array): Control inputs [thrust, roll_torque, pitch_torque, yaw_torque]
        """

        # saturate control inputs to valid range
        cntrl_values = np.clip(cntrl_values, [0.0, -0.03, -0.03, -0.03], [0.35, 0.03, 0.03, 0.03])

        # apply control inputs
        self.data.ctrl[0] = cntrl_values[0]
        self.data.ctrl[1] = cntrl_values[1]
        self.data.ctrl[2] = cntrl_values[2]
        self.data.ctrl[3] = cntrl_values[3]

    # Reading Position 
    def get_position(self) -> np.ndarray:
        """
        Get the current position of the drone in world coordinates.

        Returns:
            position (np.ndarray): Position vector [x, y, z]

        N.B 
        In the real world, the position must come from an external motion capture system or GPS 
        """
        position = self.data.qpos[0:3]
        return position
    
    # Reading Velocity
    def get_velocity(self) -> np.ndarray:
        """
        Get the current linear velocity of the drone in world coordinates.

        Returns:
            velocity (np.ndarray): Velocity vector [vx, vy, vz]
        """
        velocity = self.data.qvel[0:3]
        return velocity
    
    # Reading Sensor Methods
    def sense_angular_velocity(self) -> np.ndarray:
        """
        Read angular velocity from the IMU gyroscope sensor

        Returns:
            angular_velocity (np.ndarray): Angular velocity vector [wx, wy, wz]
        """
        angular_velocity = self.data.sensordata[self._gyro_i  : self._gyro_i  + 3]
        return angular_velocity
    
    def sense_linear_acceleration(self) -> np.ndarray:
        """
        Read linear acceleration from the IMU accelerometer sensor

        Returns:
            linear_acceleration (np.ndarray): Linear acceleration vector [ax, ay, az]
        """
        linear_acceleration = self.data.sensordata[self._accel_i : self._accel_i + 3]
        return linear_acceleration
    
    def sense_orientation(self) -> np.ndarray:
        """
        Read orientation from the IMU quaternion sensor

        Returns:
            orientation (np.ndarray): Orientation quaternion [qw, qx, qy, qz]
        """
        quat = self.data.sensordata[self._quat_i  : self._quat_i  + 4]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # reorder to (x, y, z, w)
        phi, theta, psi = r.as_euler('xyz', degrees=False)
        orientation = np.array([phi, theta, psi])
        return orientation

    """
    High-Level Drone Interface
    """

    def hover_in_place(self, target_altitude:float, verbose=False) -> None:
        """
        Command the drone to hover at a specified altitude
        - Uses a simple PID controller to maintain altitude 
        Args:
            target_altitude (float): Desired altitude to hover at (in meters).
        """

        # update target altitude for the PID controller
        self.altitude_pid.setpoint = target_altitude

        if verbose:
            print(f"Hovering at target altitude: {target_altitude} m")

        # read current altitude and compute control signal
        altitude = self.get_position()[2]
        thrust = self.altitude_pid.compute(altitude) + self.hover_bias
        cntrl = np.array([thrust, 0, 0, 0]) # thrust, no rotational torques

        # apply control
        self.set_control(cntrl)

    def move_to_position(self, target_position:np.ndarray, verbose=False) -> None:
        """
        Command the drone to move to a specified position
        - Currently only implements altitude control 
        Args:
            target_position (np.ndarray): Desired position [x, y, z] in meters.
        """

        if verbose:
            print(f"Moving to target position: {target_position} m")

        # Read current state
    
        x, y, z = self.get_position()
        x_dot, y_dot, z_dot = self.get_velocity()
        roll, pitch, yaw = self.sense_orientation()

        # Use PID to compute attitude control

        thrust = self.altitude_pid.compute(z) + self.hover_bias
        cntrl = np.array([thrust, 0, 0, 0]) # thrust, no rotational torques

        # apply control
        self.set_control(cntrl)



def main():
    drone = Drone(XML_PATH)
    print("Drone model and data initialized.")

    pid_params = [
        [0.0, 0.0, 0.0],   # pos x (disabled)
        [0.0, 0.0, 0.0],   # pos y (disabled)
        [0.8, 0.05, 0.3],  # pos z (altitude position PID)
        [0.0, 0.0, 0.0],   # vel x (disabled)
        [0.0, 0.0, 0.0],   # vel y (disabled)
        [0.8, 0.05, 0.3],  # vel z (altitude velocity PID)
        [0.0, 0.0, 0.0],   # roll torque (off)
        [0.0, 0.0, 0.0],   # pitch torque (off)
        [0.0, 0.0, 0.0],   # yaw torque (off)
    ]


    with mujoco.viewer.launch_passive(drone.model, drone.data) as viewer:
        while viewer.is_running():

            # Simple altitude hold using PID
            drone.move_to_position(target_position=np.array([0.0, 0.0, 0.2]), verbose=True)

            mujoco.mj_step(drone.model, drone.data)
            viewer.sync()
            time.sleep(0.001)

if __name__ == "__main__":
    main()