# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d

from Segmentation import Segmentation
from Registration import Registration

### Load pointcloud
pcd = o3d.io.read_point_cloud('data/box.ply')

### Segmentation
# Visualize and select region of interest
o3d.visualization.draw_geometries_with_editing([pcd])

# Construct Segmentation object
segmented_box = Segmentation(file="data/box_cropped.ply")
#segmented_box.display()

# Remove Outliers
segmented_box.removeOutliers(display=False)
#segmented_box.display()

# Compute normals
segmented_box.computeNormals()
#segmented_box.display()

# Estimate the floor's normal
segmented_box.normalsHistogram()
segmented_box.estimateFloorNormal()

# ALign the floor with the horizontal plane
segmented_box.alignFloor()
segmented_box.display()

# Remove the floor
segmented_box.removeFloor()
segmented_box.display()

o3d.io.write_point_cloud("data/box_without_floor.ply", segmented_box.pointCloud)


#new_pcd = o3d.io.read_point_cloud('data/box_without_floor.ply')
#o3d.visualization.draw_geometries_with_editing([new_pcd])



'''

### Load pointcloud
pcd2 = o3d.io.read_point_cloud('data/OtherBoxes/box2.ply')

### Segmentation
# Visualize and select region of interest
o3d.visualization.draw_geometries_with_editing([pcd2])

# Construct Segmentation object
segmented_box2 = Segmentation(file="data/box2_cropped.ply")
segmented_box.display()

# Remove Outliers
segmented_box2.removeOutliers(display=False)
segmented_box.display()

# Compute normals
segmented_box2.computeNormals()
segmented_box.display()

# Estimate the floor's normal
segmented_box2.normalsHistogram()
segmented_box2.estimateFloorNormal()

# ALign the floor with the horizontal plane
segmented_box2.alignFloor()

# Remove the floor
segmented_box2.removeFloor()
segmented_box2.display()

o3d.io.write_point_cloud("data/box2_without_floor.ply", segmented_box2.pointCloud)
'''
'''

new_pcd2 = o3d.io.read_point_cloud('data/box2_without_floor.ply')
#o3d.visualization.draw_geometries_with_editing([new_pcd2])





### Registration
# Load and visualize the objects
registration = Registration(source=new_pcd, target=new_pcd2)
registration.processGlobal()
registration.processICP(pointToPlane=True)
registration.display()

'''

# Global registration

# ICP Registration

