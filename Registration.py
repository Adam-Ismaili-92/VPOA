import copy
import open3d as o3d
import numpy as np


class Registration:
    def __init__(self, source, target, voxelSize = 0.05):
        """ 
        :param source : The source point cloud
        :param target: The target point cloud
        :param voxelSize: Size of the voxels used for downsampling
        """
        self.source = copy.deepcopy(source)
        self.target = copy.deepcopy(target)
        
        self.voxelSize = voxelSize
        
        self.result = o3d.pipelines.registration.RegistrationResult()
        
        # Downsampling
        self.sourceDown = self.source.voxel_down_sample(voxelSize)
        self.targetDown = self.target.voxel_down_sample(voxelSize)
    
        # Computing normals
        self.sourceDown.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize * 2, max_nn=30))
        self.targetDown.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize * 2, max_nn=30))
    
        # Computing fast point feature histograms
        self.sourceFpfh = o3d.pipelines.registration.compute_fpfh_feature(self.sourceDown, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize * 5, max_nn=100))
        self.targetFpfh = o3d.pipelines.registration.compute_fpfh_feature(self.targetDown, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize * 5, max_nn=100))
        
    def display(self):
        sourceTmp = copy.deepcopy(self.source)
        targetTmp = copy.deepcopy(self.target)
        sourceTmp.paint_uniform_color([1, 0.706, 0])
        targetTmp.paint_uniform_color([0, 0.651, 0.929])
        sourceTmp.transform(self.result.transformation)
        o3d.visualization.draw_geometries([sourceTmp, targetTmp])
        
    def processGlobal(self):
        """ RANSAC registration on point clouds
        """
        
        maxCorrespondanceDistance = self.voxelSize * 1.5
        # Implement o3d.pipelines.registration.registration_ransac_based_on_feature_matching() here on downsampled data
        self.result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.sourceDown, self.targetDown, self.sourceFpfh, self.targetFpfh,
            mutual_filter=True, max_correspondence_distance=maxCorrespondanceDistance)
        
        print(self.result)
        return self.result
    
    def processICP(self, pointToPlane = False):
        """ ICP registration on point clouds
        """
        
        if len(self.result.transformation) != 4:
            print('You must call processGlobal before processICP to compute an inital guess.')
            return
        
        distanceThreshold = self.voxelSize * 0.4
        
        if not pointToPlane:
            # ICP Point to Point
            self.result = o3d.pipelines.registration.registration_icp(
                self.sourceDown, self.targetDown, distanceThreshold,
                self.result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        else:
            # ICP Point to Plane
            self.result = o3d.pipelines.registration.registration_icp(
                self.sourceDown, self.targetDown, distanceThreshold,
                self.result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
        
        #self.result = o3d.pipelines.registration.evaluate_registration(self.source, self.target, 0.02, result.transformation)
        print(self.result)
        return self.result   
