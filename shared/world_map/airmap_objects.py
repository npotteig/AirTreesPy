import airsim

obstacle_info = [[50, 50, 30, 30],
                     [-50, 50, 30, 30],
                     [-50, -50, 30, 30],
                     [50, -50, 30, 30],
                     [50, 0, 10, 30],
                     [-50, 0, 10, 30],
                     [0, -50, 30, 10],
                     [0, 50, 30, 10]]

# obstacle_info = [[20, 20, 10, 30],
#                      [-50, 50, 10, 30],
#                      [-50, -50, 30, 10],
#                      [50, -50, 30, 10],
#                      [50, 0, 30, 30],
#                      [-50, 0, 30, 30],
#                      [0, -50, 30, 30],
#                      [0, 50, 30, 30]]

def spawn_walls(client, low, high, z_val):
    pos1 = airsim.Vector3r(high, 0, z_val)
    pose1 = airsim.Pose(position_val=pos1)
    pos2 = airsim.Vector3r(low, 0, z_val)
    pose2 = airsim.Pose(position_val=pos2)
    pos3 = airsim.Vector3r(0, high, z_val)
    pose3 = airsim.Pose(position_val=pos3)
    pos4 = airsim.Vector3r(0, low, z_val)
    pose4 = airsim.Pose(position_val=pos4)
    
    z_size_adj = 30
    scaleY = airsim.Vector3r(1, int(high-low), z_size_adj)
    scaleX = airsim.Vector3r(int(high-low), 1, z_size_adj)
    client.simSpawnObject('my_cube1', 'Cube', pose1, scaleY)
    client.simSpawnObject('my_cube2', 'Cube', pose2, scaleY)
    client.simSpawnObject('my_cube3', 'Cube', pose3, scaleX)
    client.simSpawnObject('my_cube4', 'Cube', pose4, scaleX)
    
def destroy_objects(client):
    objectsToDelete = ["TemplateCube", "Cone", "Cylinder", "Cube", "OrangeBall"]
    simobjects = client.simListSceneObjects()
    for so in simobjects:
        for o in objectsToDelete:
            if o in so:
                client.simDestroyObject(so)
    print("objects destroyed")

    simobjects = client.simListSceneObjects()
    for so in simobjects:
        # Sometimes objects not cleared...
        for o in objectsToDelete:
            if o in so:
                client.simDestroyObject(so)

    

def spawn_obstacles(client, z_val):
    z_size_adj = 30
    
    for i in range(len(obstacle_info)):
        obs_info = obstacle_info[i]
        pos = airsim.Vector3r(obs_info[0], obs_info[1], z_val)
        pose = airsim.Pose(position_val=pos)
        scale = airsim.Vector3r(obs_info[2], obs_info[3], z_size_adj)
        client.simSpawnObject('my_obs'+str(i), 'Cube', pose, scale)
        
    return obstacle_info
        
def inside_object(pt, obj, buf = 0):
    x = obj[0]
    y = obj[1]
    dx = obj[2] / 2
    dy = obj[3] / 2
    top_right = [x + dx, y + dy]
    bot_left = [x - dx, y - dy]
    
    if bot_left[0] - buf <= pt[0] <= top_right[0] + buf and bot_left[1] - buf <= pt[1] <= top_right[1] + buf:
        return True
    else:
        return False 
    
    
    
    