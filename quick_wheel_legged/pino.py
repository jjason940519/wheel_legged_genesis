import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt

class RobotModel:
    def __init__(self, urdf_path):
        # Load the robot model (kinematic model only)
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.pos = pin.neutral(self.model)  # Neutral joint configuration

    def get_joint_indices(self, target_joints):
        """Find joint indices for given joint names."""
        joint_indices = {}
        for joint_name in target_joints:
            joint_idx = self.model.getJointId(joint_name)
            if joint_idx < self.model.njoints:
                joint_indices[joint_name] = joint_idx - 1  # Adjust for q vector indexing
                print(f"Joint '{joint_name}' index: {joint_indices[joint_name]}")
            else:
                print(f"Joint '{joint_name}' not found or has no degrees of freedom")
        return joint_indices

    def list_all_joints(self):
        """List all joint names and their indices."""
        print("Joint names and indices:")
        for idx, name in enumerate(self.model.names[1:]):  # Skip 'universe' joint
            print(f"Index: {idx}, Joint name: {name}")

    def compute_com(self, q):
        """Compute the center of mass for a given joint configuration."""
        pin.forwardKinematics(self.model, self.data, q)
        return pin.centerOfMass(self.model, self.data, q)

    def get_joint_position(self, q, joint_name, base_link_name="base_link"):
        """Get the position of a joint relative to base_link frame."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get joint and base_link frame IDs
        joint_frame_id = self.model.getFrameId(joint_name)
        base_frame_id = self.model.getFrameId(base_link_name)
        if joint_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame '{joint_name}' not found in URDF")
        if base_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame '{base_link_name}' not found in URDF")

        # Get joint position in world frame
        joint_position_world = self.data.oMf[joint_frame_id].translation
        # Get transformation from world to base_link
        base_frame_inv = self.data.oMf[base_frame_id].inverse()  # base_link to world
        # Transform joint position to base_link frame
        joint_position_homogeneous = np.array([joint_position_world[0], joint_position_world[1], joint_position_world[2], 1.0])
        joint_position_base_link = base_frame_inv.homogeneous @ joint_position_homogeneous
        return joint_position_base_link[:3]  # Return x, y, z

    def transform_com_to_base_link(self, q, com, base_link_name="base_link"):
        """Transform CoM from world frame to base_link frame."""
        # Update frame placements
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get base_link frame ID
        base_frame_id = self.model.getFrameId(base_link_name)
        if base_frame_id >= len(self.model.frames):
            raise ValueError(f"Frame '{base_link_name}' not found in URDF")

        # Get transformation from world to base_link
        base_frame_placement = self.data.oMf[base_frame_id]  # SE3 object (world to base_link)
        base_frame_inv = base_frame_placement.inverse()  # base_link to world

        # Transform CoM to base_link frame
        com_homogeneous = np.array([com[0], com[1], com[2], 1.0])  # CoM in homogeneous coordinates
        com_base_link = base_frame_inv.homogeneous @ com_homogeneous  # Transform to base_link frame
        return com_base_link[:3]  # Return x, y, z (ignore homogeneous coordinate)

    def analyze_com_over_angle_ranges(self, joint_indices, hip_angle_range, thigh_angle_range, base_link_name="base_link"):
        """Analyze CoM over a range of hip and thigh angles, transform to base_link frame."""
        hip_angles = np.linspace(np.deg2rad(hip_angle_range[0]), np.deg2rad(hip_angle_range[1]), 5)
        thigh_angles = np.linspace(np.deg2rad(thigh_angle_range[0]), np.deg2rad(thigh_angle_range[1]), 5)
        com_positions_base_link = []
        for hip_angle in hip_angles:
            for thigh_angle in thigh_angles:
                q = self.pos.copy()  # Start from neutral configuration
                q[joint_indices["L_thigh_joint"]] = hip_angle
                q[joint_indices["R_thigh_joint"]] = hip_angle
                q[joint_indices["L_calf_joint"]] = thigh_angle
                q[joint_indices["R_calf_joint"]] = thigh_angle
                com = self.compute_com(q)
                com_base_link = self.transform_com_to_base_link(q, com, base_link_name)
                com_positions_base_link.append(com_base_link)
                print(f"Hip angle: {np.rad2deg(hip_angle):.2f}°, Thigh angle: {np.rad2deg(thigh_angle):.2f}°, "
                      f"CoM in base_link (x, y, z): {com_base_link}")
        return com_positions_base_link

    def plot_com_2d(self, com_positions, leg_pos):
        """Plot CoM positions in 2D (x-z plane) using Matplotlib."""
        # Extract x and z coordinates from com_positions (ignore y)
        com_positions = np.array(com_positions)
        x = com_positions[:, 0]
        z = com_positions[:, 2]

        # Create 2D scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x, z, c='b', marker='o', label='CoM Positions')
        plt.plot(0, 0, 'r*', markersize=15, label='base_link Origin (0,0)')  # Mark origin

        # Set labels and title
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Center of Mass Positions in base_link Frame (X-Z Plane)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Equal scaling for x and z axes

        # Show plot
        plt.show()

# Usage
if __name__ == "__main__":
    # Initialize the robot model
    urdf_path = "../assets/quick_bipedal_urdf/urdf/quick_bipedal.urdf"
    robot = RobotModel(urdf_path) 

    target_joints = ["L_thigh_joint", "R_thigh_joint", "L_calf_joint", "R_calf_joint", "L_wheel_joint", "R_wheel_joint"]  # Example names
    joint_indices = robot.get_joint_indices(target_joints)


    # List all joints to identify hip and thigh joint names
    robot.list_all_joints()


    hip_angle_range = [0, 90]
    thigh_angle_range = [-180, 0]  # Example: [-90°, 90°]
    com_positions_base_link = robot.analyze_com_over_angle_ranges(joint_indices, hip_angle_range, thigh_angle_range)

    # Plot the CoM positions in 2D (x-z plane)
    robot.plot_com_2d(com_positions_base_link)