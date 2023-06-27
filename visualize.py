from utils import load_cameras
import numpy as np
import trimesh


def check_rot(rot, right_handed=True, eps=1e-6):
    """
    Input: 3x3 rotation matrix
    """
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3

    assert np.allclose(rot.transpose() @ rot, np.eye(3), atol=1e-6)
    assert np.linalg.det(rot) - 1 < eps * 2

    if right_handed:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) - 1.0) < 1e-3
    else:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) + 1.0) < 1e-3

def visualize_cameras(rotation_matrices, camera_positions, line_length=1.0, obj_f = None):
    # Define the camera axes directions in the camera coordinate system
    axes_directions = np.array([[1, 0, 0], 
                                [0, 1, 0], 
                                [0, 0, -1]])

    # Create a trimesh scene
    scene = trimesh.Scene()

    if obj_f is not None:
        mesh = trimesh.load_mesh(obj_f)
        # mesh.apply_scale(0.0155523)
        scene.add_geometry(mesh)

    # Iterate over the rotation matrices and camera positions
    for i in range(len(rotation_matrices)):
        rotation_matrix = rotation_matrices[i].T
        right, up, forward = rotation_matrix
        rotation_matrix = np.stack([right, up, forward])
        camera_position = np.array(camera_positions[i])

        # check_rot(rotation_matrix, right_handed=True)

        # Transform the axes directions to the world coordinate system
        # world_axes = np.dot(rotation_matrix, axes_directions.T).T
        world_axes = rotation_matrix@axes_directions

        # Define the colors for each axis
        # colors = ['r', 'g', 'b']
        # colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]

        # Add the camera axes to the scene as line segments with corresponding colors
        for j, axis in enumerate(world_axes):
            # Calculate the endpoint of the axis
            if j == 2:  # Z-axis
                endpoint = camera_position - 80 * line_length * axis
            else:
                endpoint = camera_position + line_length * axis

            # Create a line segment geometry with the corresponding color
            line = trimesh.load_path([[camera_position, endpoint]], colors=[colors[j]])
            # line = trimesh.load_path([[camera_position, endpoint]], colors=[(0,0,0,0)])

            # Add the line segment to the scene
            scene.add_geometry(line)


    # Show the scene
    scene.show()



def main():
    gap = 1
    # cam_dir = "data/robo_cameras"
    # obj_f = "data/mecha_cropped.glb"

    # cam_dir = "data/carpet_cameras"
    obj_f = "flying_room/flying_room.obj"

    cam_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/ShakeCarpet1_formatted/ecam_set/camera"
    rots, trans = load_cameras(cam_dir)
    rots, trans = rots[::gap], trans[::gap]

    # visualize_cameras(rots, trans, 0.1, obj_f="flying_room/flying_room.obj")
    visualize_cameras(rots, trans, 0.1, obj_f=obj_f)


if __name__ == "__main__":
    main()
