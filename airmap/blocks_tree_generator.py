import airsim
import pprint
from std_msgs.msg import String, Float32MultiArray
import os
import numpy as np
import time
import random
import csv

obstacle_info = []

def build_blocks_world(client, load=False):
    vehicle_name = "Drone1"
    client = airsim.MultirotorClient()
    client.confirmConnection()

    ball_heading = np.random.normal(0, 2)
    objectsToDelete = ["TemplateCube", "Cone", "Cylinder", "Cube", "OrangeBall"]
    simobjects = client.simListSceneObjects()
    for so in simobjects:
        for o in objectsToDelete:
            if o in so:
                client.simDestroyObject(so)
                # print(str(so)+" destroyed")
    print("objects destroyed")

    # print("Remaining objects:")
    simobjects = client.simListSceneObjects()
    for so in simobjects:
        # print(str (so))
        # Sometimes objects not cleared...
        for o in objectsToDelete:
            if o in so:
                client.simDestroyObject(so)
                # print(str(so)+" destroyed")
            # else:
            #     # Debug - what's left:
            #     print(so)

    dronePosition = client.simGetGroundTruthEnvironment(vehicle_name).position

    targetPose = airsim.Pose()
    targetPose.position.x_val = 10
    targetPose.position.y_val = 0
    targetPose.position.z_val = -1
    while getDistance(dronePosition, targetPose.position) < 10:
        targetPose.position.x_val = random.randint(-50, 50)
        targetPose.position.y_val = random.randint(-50, 50)
    TargetName = "OrangeBall_0"
    # scale = airsim.Vector3r(2, 2, 2)   
    # self.client.simSpawnObject(self.TargetName, 'Sphere', targetPose, scale, physics_enabled=True, is_blueprint=False)
    scale = airsim.Vector3r(0.50, 0.50, 1.8)
    client.simSpawnObject(TargetName, 'Cylinder', targetPose, scale, physics_enabled=True, is_blueprint=False)
    client.simSetSegmentationObjectID(TargetName, 255)
    print(client.simSetObjectMaterialFromTexture(TargetName, "C:/Users/Daniel/Downloads/Airsim/Blocks/WindowsNoEditor/orange.jpg"))

    obstacles = []

    pose = airsim.Pose()
    
    fileDir = os.path.dirname(os.path.abspath(__file__)) + '/obstacles/'
    
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
        
    fileName = fileDir +'obstacles.txt'
    
    if load:
        with open(fileName, mode='r') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=",", quotechar='"')
            for row in csvReader:
                pose.position.x_val = float(row[1])
                pose.position.y_val = float(row[2])
                pose.position.z_val = float(row[3])
                width = float(row[4])
                height = float(row[5])
                scale = airsim.Vector3r(width, width, height)
                if -300 <= float(row[1]) <= 300 and -300 <= float(row[2]) <= 300:
                    obstacle_info.append([float(row[1]), float(row[2]),
                                        float(row[4]), float(row[4])])
                    CylinderName = row[0]                   
                    client.simSpawnObject(CylinderName, 'Cylinder', pose, scale, physics_enabled=False, is_blueprint=False)
                    client.simSetSegmentationObjectID(CylinderName, random.randint(0,250))
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
            if getPlanarDistance(dronePosition, pose.position) > width and getPlanarDistance(targetPose.position, pose.position) > width:
                # Crown
                client.simSpawnObject(CylinderName, 'Cylinder', pose, scale, physics_enabled=False, is_blueprint=False)
                client.simSetSegmentationObjectID(CylinderName, random.randint(0,250))
                client.simSetObjectMaterialFromTexture(CylinderName, "C:/Users/Daniel/Downloads/Airsim/Blocks/WindowsNoEditor/green.jpg")
                obstacles.append([CylinderName, pose.position.x_val, pose.position.y_val, pose.position.z_val, width, height])
                # Trunk
                CylinderName = "Cylinder_"+str(i)+"_trunk"
                scale = airsim.Vector3r(width/10, width/10, height)
                pose.position.z_val = -height/2 + 1
                client.simSpawnObject(CylinderName, 'Cylinder', pose, scale, physics_enabled=False, is_blueprint=False)
                client.simSetSegmentationObjectID(CylinderName, random.randint(0,250))
                obstacles.append([CylinderName, pose.position.x_val, pose.position.y_val, pose.position.z_val, width/10, height])
            else:
                print("skipping spawning object due to unsafe distance to drone or ball")
        print("%d objects spawned" %len(obstacles))
        
        

        
        with open(fileName, mode='w') as csvFile:
            csvWriter = csv.writer(csvFile, delimiter=",", quotechar='"')
            csvWriter.writerows(obstacles)
            print("Obstacles stored in: %s" % fileName)


def getDistance(object_a_position, object_b_position):
    p1 = np.array([object_a_position.x_val, object_a_position.y_val, object_a_position.z_val])
    p2 = np.array([object_b_position.x_val, object_b_position.y_val, object_b_position.z_val])    
    return np.sqrt(np.sum((p1-p2)**2, axis=0))

def getPlanarDistance(object_a_position, object_b_position):
    p1 = np.array([object_a_position.x_val, object_a_position.y_val])
    p2 = np.array([object_b_position.x_val, object_b_position.y_val])    

    return np.sqrt(np.sum((p1-p2)**2, axis=0))

