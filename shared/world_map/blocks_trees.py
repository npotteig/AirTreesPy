import os
import random
import csv

import airsim

from shared.world_map.world_map import Map

# transfer environment
# Forest of low-poly trees

class BlocksTrees(Map):
    def __init__(self, client: airsim.client, load: str = None, load_bounds:list=[-300, 300]) -> None:
        super().__init__(client, load)
        
        self.load_bounds = load_bounds
    
    def build_world(self) -> None:
        vehicle_name = "Drone1"
        self.destroy_objects()
        
        dronePosition = self.client.simGetGroundTruthEnvironment(vehicle_name).position

        targetPose = airsim.Pose()
        targetPose.position.x_val = 10
        targetPose.position.y_val = 0
        targetPose.position.z_val = -1
        while self.getDistance(dronePosition, targetPose.position) < 10:
            targetPose.position.x_val = random.randint(-50, 50)
            targetPose.position.y_val = random.randint(-50, 50)
        TargetName = "OrangeBall_0"
        # scale = airsim.Vector3r(2, 2, 2)   
        # self.client.simSpawnObject(self.TargetName, 'Sphere', targetPose, scale, physics_enabled=True, is_blueprint=False)
        scale = airsim.Vector3r(0.50, 0.50, 1.8)
        self.client.simSpawnObject(TargetName, 'Cylinder', targetPose, scale, physics_enabled=True, is_blueprint=False)
        self.client.simSetSegmentationObjectID(TargetName, 255)
        print(self.client.simSetObjectMaterialFromTexture(TargetName, "C:/Users/Daniel/Downloads/Airsim/Blocks/WindowsNoEditor/orange.jpg"))

        obstacles = []

        pose = airsim.Pose()
        
        fileDir = os.path.dirname(os.path.abspath(__file__)) + '/tree_obstacles/'
        
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)
            
        fileName = fileDir +'obstacles.txt'
        
        if self.load:
            with open(fileName, mode='r') as csvFile:
                csvReader = csv.reader(csvFile, delimiter=",", quotechar='"')
                for row in csvReader:
                    pose.position.x_val = float(row[1])
                    pose.position.y_val = float(row[2])
                    pose.position.z_val = float(row[3])
                    width = float(row[4])
                    height = float(row[5])
                    scale = airsim.Vector3r(width, width, height)
                    if self.load_bounds[0] <= float(row[1]) <= self.load_bounds[1] and self.load_bounds[0] <= float(row[2]) <= self.load_bounds[1]:
                        self.obstacle_info.append([float(row[1]), float(row[2]),
                                            float(row[4]), float(row[4])])
                        CylinderName = row[0]                   
                        self.client.simSpawnObject(CylinderName, 'Cylinder', pose, scale, physics_enabled=False, is_blueprint=False)
                        self.client.simSetSegmentationObjectID(CylinderName, random.randint(0,250))
            print("Objects spawned")
            # print(obstacle_info)
        
        else:
        
            # meters
            obstacleWidthMin = 10
            obstacleWidthMax = 50
            obstacleHeightMin = 5
            obstacleHeightMax = 30

            for i in range(1000):
                # 5 sqmi in 50x50m raster
                pose.position.x_val = float(random.randint(-36, 36)*50)
                pose.position.y_val = float(random.randint(-36, 36)*50)
                width = random.randint(obstacleWidthMin, obstacleWidthMax)
                height = random.randint(obstacleHeightMin, obstacleHeightMax)
                pose.position.z_val = -height/2 - random.randint(1, 5) 
                scale = airsim.Vector3r(width, width, height)
                CylinderName = "Cylinder_"+str(i)+"_crown" 
                if self.getPlanarDistance(dronePosition, pose.position) > width and self.getPlanarDistance(targetPose.position, pose.position) > width:
                    # Crown
                    self.client.simSpawnObject(CylinderName, 'Cylinder', pose, scale, physics_enabled=False, is_blueprint=False)
                    self.client.simSetSegmentationObjectID(CylinderName, random.randint(0,250))
                    self.client.simSetObjectMaterialFromTexture(CylinderName, "C:/Users/Daniel/Downloads/Airsim/Blocks/WindowsNoEditor/green.jpg")
                    obstacles.append([CylinderName, pose.position.x_val, pose.position.y_val, pose.position.z_val, width, height])
                    # Trunk
                    CylinderName = "Cylinder_"+str(i)+"_trunk"
                    scale = airsim.Vector3r(width/10, width/10, height)
                    pose.position.z_val = -height/2 + 1
                    self.client.simSpawnObject(CylinderName, 'Cylinder', pose, scale, physics_enabled=False, is_blueprint=False)
                    self.client.simSetSegmentationObjectID(CylinderName, random.randint(0,250))
                    obstacles.append([CylinderName, pose.position.x_val, pose.position.y_val, pose.position.z_val, width/10, height])
                else:
                    print("skipping spawning object due to unsafe distance to drone or ball")
            print("%d objects spawned" %len(obstacles))
            
            

            
            with open(fileName, mode='w') as csvFile:
                csvWriter = csv.writer(csvFile, delimiter=",", quotechar='"')
                csvWriter.writerows(obstacles)
                print("Obstacles stored in: %s" % fileName)