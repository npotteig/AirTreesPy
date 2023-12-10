import airsim

from shared.world_map.world_map import Map

# training environment
# Simple Blocks Maze

class BlocksMaze(Map):
    def __init__(self, client: airsim.client, load: str=None) -> None:
        super().__init__(client, load)
        
        self.obstacle_info = [[50, 50, 30, 30],
                     [-50, 50, 30, 30],
                     [-50, -50, 30, 30],
                     [50, -50, 30, 30],
                     [50, 0, 10, 30],
                     [-50, 0, 10, 30],
                     [0, -50, 30, 10],
                     [0, 50, 30, 10]]
        
        self.z_size_adj = 30
        self.z_val = -32
        self.wall_low = -200
        self.wall_high = 200
        
    def build_world(self) -> None:
        self.destroy_objects()
        self.spawn_walls()
        self.spawn_obstacles()
    
    def spawn_walls(self):
        pos1 = airsim.Vector3r(self.wall_high, 0, self.z_val)
        pose1 = airsim.Pose(position_val=pos1)
        pos2 = airsim.Vector3r(self.wall_low, 0, self.z_val)
        pose2 = airsim.Pose(position_val=pos2)
        pos3 = airsim.Vector3r(0, self.wall_high, self.z_val)
        pose3 = airsim.Pose(position_val=pos3)
        pos4 = airsim.Vector3r(0, self.wall_low, self.z_val)
        pose4 = airsim.Pose(position_val=pos4)
        
        self.z_adj_size = 30
        scaleY = airsim.Vector3r(1, int(self.wall_high-self.wall_low), self.z_size_adj)
        scaleX = airsim.Vector3r(int(self.wall_high-self.wall_low), 1, self.z_size_adj)
        self.client.simSpawnObject('my_cube1', 'Cube', pose1, scaleY)
        self.client.simSpawnObject('my_cube2', 'Cube', pose2, scaleY)
        self.client.simSpawnObject('my_cube3', 'Cube', pose3, scaleX)
        self.client.simSpawnObject('my_cube4', 'Cube', pose4, scaleX) 
    
    def spawn_obstacles(self):
        for i in range(len(self.obstacle_info)):
            obs_info = self.obstacle_info[i]
            pos = airsim.Vector3r(obs_info[0], obs_info[1], self.z_val)
            pose = airsim.Pose(position_val=pos)
            scale = airsim.Vector3r(obs_info[2], obs_info[3], self.z_size_adj)
            self.client.simSpawnObject('my_obs'+str(i), 'Cube', pose, scale)
            