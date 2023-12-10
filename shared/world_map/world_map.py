import airsim
from abc import abstractmethod
import numpy as np

class Map(object):
    """Map Base Class
    """
    
    def __init__(self, client: airsim.client, load: str=None) -> None:
        self.client = client
        self.load = load
        self.obstacle_info = []
    
    @abstractmethod
    def build_world(self) -> None:
        pass
    
    def inside_objects(self, pt, buf = 0):
        """Checks if pt is inside any object in self.obstacles_info

        Args:
            pt (_type_): _description_
        """
        for obstacle in self.obstacle_info:
            if self.inside_object(pt, obstacle, buf):
                return True
        return False
    
    def inside_object(self, pt, obj, buf = 0):
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
    
    def getDistance(self, object_a_position, object_b_position):
        p1 = np.array([object_a_position.x_val, object_a_position.y_val, object_a_position.z_val])
        p2 = np.array([object_b_position.x_val, object_b_position.y_val, object_b_position.z_val])    
        return np.sqrt(np.sum((p1-p2)**2, axis=0))

    def getPlanarDistance(self, object_a_position, object_b_position):
        p1 = np.array([object_a_position.x_val, object_a_position.y_val])
        p2 = np.array([object_b_position.x_val, object_b_position.y_val])    

        return np.sqrt(np.sum((p1-p2)**2, axis=0))
    
    def destroy_objects(self):
        objectsToDelete = ["TemplateCube", "Cone", "Cylinder", "Cube", "OrangeBall"]
        simobjects = self.client.simListSceneObjects()
        for so in simobjects:
            for o in objectsToDelete:
                if o in so:
                    self.client.simDestroyObject(so)
        print("objects destroyed")

        simobjects = self.client.simListSceneObjects()
        for so in simobjects:
            # Sometimes objects not cleared...
            for o in objectsToDelete:
                if o in so:
                    self.client.simDestroyObject(so)