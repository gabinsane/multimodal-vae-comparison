from lanro_gym.simulation import PyBulletSimulation

PLANE_COLOR = [1, 1, 1, 1]
TABLE_COLOR = [1, 1, 1, 1]


def basic_scene(
    sim: PyBulletSimulation,
    plane_z_offset: float = -0.4,
    plane_x_pos: float = -0.2,
    plane_length: float = 0.8,
    table_length: float = 1.8,
    table_width: float = 1.8,
    table_height: float = 0.4,
    table_x_offset: float = -0.1,
    table_z_offset: float = 0.0,
    table_friction: float = 0.2,
    table_spinning_friction: float = 0.000,
):
    sim.create_box(
        body_name="plane",
        half_extents=[plane_length, 0.8, 0.01],
        mass=0,
        position=[plane_x_pos, 0.0, plane_z_offset - 0.01],
        specular_color=[0.0, 0.0, 0.0],
        rgba_color=PLANE_COLOR,
    )
    sim.create_box(
        body_name="table",
        half_extents=[table_length / 2, table_width / 2, table_height / 2],
        mass=0,
        position=[table_x_offset, 0.0, -table_height / 2 + table_z_offset],
        specular_color=[0.0, 0.0, 0.0],
        rgba_color=TABLE_COLOR,
        lateral_friction=table_friction,
        spinning_friction=table_spinning_friction,
    )
    sim.create_box(
        body_name="robot_platform",
        half_extents=[0.175, 0.175, 0.2],
        mass=0,
        position=[-0.675, 0.0, -0.2],
        specular_color=[0.0, 0.0, 0.0],
        rgba_color=TABLE_COLOR,
        lateral_friction=table_friction,
    )
