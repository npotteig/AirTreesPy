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
    
def destroy_walls(client):
    client.simDestroyObject('my_cube1')
    client.simDestroyObject('my_cube2')
    client.simDestroyObject('my_cube3')
    client.simDestroyObject('my_cube4')

def spawn_obstacles(client, z_val):
    z_size_adj = 5
    pos1 = airsim.Vector3r(50, 50, z_val)
    obs1 = airsim.Pose(position_val=pos1)
    pos2 = airsim.Vector3r(-50, 50, z_val)
    obs2 = airsim.Pose(position_val=pos2)
    pos3 = airsim.Vector3r(-50, -50, z_val)
    obs3 = airsim.Pose(position_val=pos3)
    pos4 = airsim.Vector3r(50, -50, z_val)
    obs4 = airsim.Pose(position_val=pos4)
    
    pos5 = airsim.Vector3r(50, 0, z_val)
    obs5 = airsim.Pose(position_val=pos5)
    pos6 = airsim.Vector3r(-50, 0, z_val)
    obs6 = airsim.Pose(position_val=pos6)
    pos7 = airsim.Vector3r(0, -50, z_val)
    obs7 = airsim.Pose(position_val=pos7)
    pos8 = airsim.Vector3r(0, 50, z_val)
    obs8 = airsim.Pose(position_val=pos8)
    
    scale = airsim.Vector3r(30, 30, z_size_adj)
    scX = airsim.Vector3r(30, 10, z_size_adj)
    scY = airsim.Vector3r(10, 30, z_size_adj)
    client.simSpawnObject('my_obs1', 'Cube', obs1, scale)
    client.simSpawnObject('my_obs2', 'Cube', obs2, scale)
    client.simSpawnObject('my_obs3', 'Cube', obs3, scale)
    client.simSpawnObject('my_obs4', 'Cube', obs4, scale)
    
    client.simSpawnObject('my_obs5', 'Cube', obs5, scY)
    client.simSpawnObject('my_obs6', 'Cube', obs6, scY)
    client.simSpawnObject('my_obs7', 'Cube', obs7, scX)
    client.simSpawnObject('my_obs8', 'Cube', obs8, scX)
    
    