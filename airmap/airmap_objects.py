import airsim

def spawn_walls(client, low, high, z_val):
    pos1 = airsim.Vector3r(high, 0, z_val)
    pose1 = airsim.Pose(position_val=pos1)
    pos2 = airsim.Vector3r(low, 0, z_val)
    pose2 = airsim.Pose(position_val=pos2)
    pos3 = airsim.Vector3r(0, high, z_val)
    pose3 = airsim.Pose(position_val=pos3)
    pos4 = airsim.Vector3r(0, low, z_val)
    pose4 = airsim.Pose(position_val=pos4)
    
    z_size_adj = 5
    scaleY = airsim.Vector3r(1, int(high-low), z_size_adj)
    scaleX = airsim.Vector3r(int(high-low), 1, z_size_adj)
    client.simSpawnObject('my_cube1', 'Cube', pose1, scaleY)
    client.simSpawnObject('my_cube2', 'Cube', pose2, scaleY)
    client.simSpawnObject('my_cube3', 'Cube', pose3, scaleX)
    client.simSpawnObject('my_cube4', 'Cube', pose4, scaleX)