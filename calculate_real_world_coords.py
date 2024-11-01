import numpy as np
def calculate_real_world_coords(camera_matrix, image_width, image_height, camera_height, roll_angle, pitch_angle, yaw_angle):
    # Convert angles to radians
    roll = np.radians(roll_angle)
    pitch = np.radians(pitch_angle)
    yaw = np.radians(yaw_angle)

    # Extract focal length from camera matrix
    focal_length_pixels = camera_matrix[0, 0]  # Assuming fx = fy

    # Rotation matrices for roll, pitch, and yaw
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll

    # Calculate the projection matrix
    K = np.array([[focal_length_pixels, 0, image_width / 2],
                  [0, focal_length_pixels, image_height / 2],
                  [0, 0, 1]])

    # Create arrays for x and y coordinates
    x_coords, y_coords = np.meshgrid(np.arange(image_width), np.arange(image_height))
    ones = np.ones_like(x_coords)

    # Convert pixel coordinates to homogeneous coordinates
    pixel_coords = np.stack([x_coords, y_coords, ones], axis=-1).reshape(-1, 3).T

    # Apply rotation to pixel coordinates
    rotated_coords = R @ np.linalg.inv(K) @ pixel_coords

    # Project onto ground plane (z = -camera_height)
    scale = -camera_height / rotated_coords[2, :]
    x_world = rotated_coords[0, :] * scale
    y_world = rotated_coords[1, :] * scale

    return x_world, y_world