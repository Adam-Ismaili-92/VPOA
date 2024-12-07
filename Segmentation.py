# -*- coding: utf-8 -*-
import copy
import math
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def rotationMatrixFromVectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    # Find and implement a solution here
    # Normalize the input vectors to ensure they have unit length
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    # Calculate the cross product of vec1 and vec2
    cross_product = np.cross(vec1, vec2)

    # Calculate the dot product of vec1 and vec2
    dot_product = np.dot(vec1, vec2)

    # Create the skew-symmetric cross product matrix
    cross_product_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                    [cross_product[2], 0, -cross_product[0]],
                                    [-cross_product[1], cross_product[0], 0]])

    # Calculate the rotation matrix using the Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + cross_product_matrix + np.dot(cross_product_matrix, cross_product_matrix) * (1 / (1 + dot_product))

    return rotation_matrix

class Segmentation:
    def __init__(self, file = 'data/cropped_1.ply'):
        self.pointCloud = o3d.io.read_point_cloud(file)
    
    def removeOutliers(self, display = False):
        # Implement outliers removal here (remove_statistical_outlier)
        cl, ind = self.pointCloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.8)
        
        inlierCloud = self.pointCloud.select_by_index(ind) # Select inlier points here (see select_by_index())
        outlierCloud = self.pointCloud.select_by_index(ind, invert=True) # Select outlier points here (see select_by_index())
        
        if display:
            box = self.pointCloud.get_axis_aligned_bounding_box()
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame( size=0.05, origin=np.asarray(box.get_box_points())[0])
            outlierCloud.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([inlierCloud, outlierCloud, box, frame])   
        
        self.pointCloud = inlierCloud
        
    def computeNormals(self, normalize = True, alignVector = [] ):
        # Implement estimate_normals() here with an Hybrid KD-Tree Search method
        self.pointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        alignVector = np.asarray(alignVector)
        if alignVector.size == 3:
            # Implement normals alignement here
            alignVector = alignVector / np.linalg.norm(alignVector)  # Normalisation du vecteur d'alignement
            self.pointCloud.orient_normals_towards_camera_location(lookat=alignVector)
            
        if normalize:
            # Implement normals normalization here
            self.pointCloud.normalize_normals()
            
        self.normals = np.asarray(self.pointCloud.normals)
        return self.normals
            
    def estimateFloorNormal(self, bins = 10):
        histX, edgesX = np.histogram(self.normals[:,0],  bins = bins)
        histY, edgesY = np.histogram(self.normals[:,1], bins = bins)
        histZ, edgesZ = np.histogram(self.normals[:,2], bins = bins)
        
        minX, maxX = (edgesX[np.argmax(histX)], edgesX[np.argmax(histX)+1])
        minY, maxY = (edgesY[np.argmax(histY)], edgesY[np.argmax(histY)+1])
        minZ, maxZ = (edgesZ[np.argmax(histZ)], edgesZ[np.argmax(histZ)+1])

        floorNormals = np.empty((0, 3))
        for line in self.normals:
            if minX <= line[0] <= maxX:       
                if minY <= line[1] <= maxY:
                    if minZ <= line[2] <= maxZ:
                        floorNormals = np.append(floorNormals, [line], axis = 0)
        
        self.floorNormal = floorNormals.mean(axis=0)
        return self.floorNormal
    
    def alignFloor(self):
        # Align the floor with the horizontal plane here
        if self.floorNormal is not None:
            # Vecteur de direction du sol vers le haut
            upward_vector = np.array([0, 0, -1])
            
            # Calculez la matrice de rotation pour aligner le vecteur normal du sol avec la direction vers le haut
            rotation_matrix = rotationMatrixFromVectors(self.floorNormal, upward_vector)
            
            # Translation pour déplacer l'origine sur le sol
            translation = -np.dot(self.pointCloud.get_center(), self.floorNormal) * self.floorNormal
            
            # Appliquez la matrice de rotation et la translation aux normales dans le nuage de points
            self.pointCloud.rotate(rotation_matrix, center=(0, 0, 0))
            self.pointCloud.translate(translation)

            # Mettez à jour les normales après l'alignement
            self.pointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            self.pointCloud.normalize_normals()
            self.normals = np.asarray(self.pointCloud.normals)
                        
    def removeFloor(self, threshold=0.55):
        xyz = np.asarray(self.pointCloud.points)
        newXYZ = np.empty((0, 3))

        for point in xyz:
            # Calculez la distance du point au plan du sol
            distance_to_floor = np.abs(np.dot(point, self.floorNormal))
            
            # Si la distance est supérieure au seuil, ajoutez le point au nouveau nuage de points
            if distance_to_floor > threshold:
                newXYZ = np.append(newXYZ, [point], axis=0)
                
        self.pointCloud.points = o3d.utility.Vector3dVector(newXYZ)
        
    def display(self, edit=False):
        if not edit:
            box = self.pointCloud.get_axis_aligned_bounding_box()
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame( size=0.05, origin=np.asarray(box.get_box_points())[0])
            o3d.visualization.draw_geometries([self.pointCloud, box, frame])
        else:
             o3d.visualization.draw_geometries_with_editing([self.pointCloud])
        
    def normalsHistogram(self, bins = 20):
        fig, ax = plt.subplots(1, 3, sharey=True, tight_layout=True)
        ax[0].set_title('X axis Hist')
        ax[1].set_title('Y axis Hist')
        ax[2].set_title('Z axis Hist')
        
        ax[0].hist(self.normals[:,0], bins= bins)
        ax[1].hist(self.normals[:,1], bins= bins)
        ax[2].hist(self.normals[:,2], bins= bins)
        plt.show()

        
    
