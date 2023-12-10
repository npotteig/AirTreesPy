import random
import os

import airsim
import numpy as np

from shared.world_map.world_map import Map

class Neighborhood(Map):
    def __init__(self, client: airsim.client, load: str = None) -> None:
        super().__init__(client, load)
    
    def build_world(self) -> None:
        objects = self.client.simListSceneObjects()
        car_poses = []
        for o in objects:
            if 'Car_' in o and 'Porch' not in o:
                car_poses.append(self.client.simGetObjectPose(o))
                if self.client.simDestroyObject(o):
                    print(o + " deleted")
        

        random.shuffle(car_poses)

        TargetName = "Car_Target_"+str(np.random.randint(1000))
        scale = airsim.Vector3r(1, 1, 1)
        if len(car_poses) > 0:
            if self.client.simSpawnObject(TargetName, 'Car_01', car_poses[0], scale, physics_enabled=False, is_blueprint=False):
                print(car_poses[0].position)
            self.client.simSetSegmentationObjectID(TargetName, 0)
            if len(os.environ.get("WSL_HOST_IP"))>0:
                if self.client.simSetObjectMaterialFromTexture(TargetName, os.environ.get("WINDOWS_AIRSIM_HOME")+"/airsim-ros2/ros_ws/src/ansr/ansr/red.png"):
                    print("Target color changed to red")
            else:
                if self.client.simSetObjectMaterialFromTexture(TargetName, "/home/airsim-ros2/ros_ws/src/ansr/ansr/red.png"):
                    print("Target color changed to red")
                

            for i in range(len(car_poses)-1):
                targetPose = car_poses[i+1]
                # targetPose.position.x_val = np.random.randint(-50,50)
                # targetPose.position.y_val = np.random.randint(-50,50)
                # targetPose.orientation = airsim.to_quaternion(
                #     np.deg2rad(0), #p
                #     np.deg2rad(0), #r
                #     np.deg2rad(np.random.randint(359)) #y
                # )
                car_name = "Car_Additional_"+str(np.random.randint(1000))
                self.client.simSpawnObject(car_name, 'Car_01', targetPose, scale, physics_enabled=False, is_blueprint=False)
                self.client.simSetSegmentationObjectID(car_name, 254)
                print(str(i) + ". vehicle spawned")

        # print(client.simSetObjectMaterial(TargetName, "M_Truck_01.MI_Car_04"))
        # # print('simSwapTextures' + str(client.simSwapTextures(TargetName, 2)))