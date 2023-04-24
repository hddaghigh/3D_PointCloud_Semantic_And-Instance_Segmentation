#!/usr/bin/env python
# coding: utf-8

# In[868]:


#from pyntcloud import PyntCloud 
import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys
import pdb
# import pclpy
# from pclpy import pcl
# import pcl
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from auxiliary_functions_Dec_27_2020_testing import *
# import pcl.pcl_visualization
import math
import time
import sklearn.preprocessing

tic1 = time.time() 

#####Path
#out_path="F:\\edge and plane extraction\\final_codes_Plane detection\\final_codes-05-08-2020-1214am\\Output\\"
pcd = o3d.geometry.PointCloud()
pcd = o3d.io.read_point_cloud("colorado_1013k_points.pcd")

tic2 = time.time() 
print("Data loading time (sec):", (tic2-tic1))


# In[204]:


#o3d.visualization.draw_geometries([pcd])


# In[869]:

'''
uni_down_pcd = pcd.uniform_down_sample(every_k_points=10)
o3d.visualization.draw_geometries([uni_down_pcd])
'''

# In[221]:


print(np.asarray(pcd.points).shape)
print(np.asarray(pcd.colors).shape)


# In[500]:
'''

import pptk
uni_down_pcd_2 = pcd.uniform_down_sample(every_k_points=20)
v = pptk.viewer(uni_down_pcd_2.points)
v.set(point_size=0.01)

'''
# ## MLS

# In[223]:

'''
# MLS
import pclpy
pcd_np = np.asarray(pcd.points)
pcd_new = pclpy.pcl.PointCloud.PointXYZ.from_array(pcd_np)
output = pcd_new.moving_least_squares(search_radius=0.02, compute_normals=False, num_threads=8)
print(np.asarray(output.points).shape)

'''

# In[ ]:





# In[873]:


downpcd = pcd.voxel_down_sample(voxel_size = 0.02)
downpcd = pcd

# In[874]:

radius = 0.3
max_nn = 100
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
Normal_Vec_allpoints = np.asarray(downpcd.normals)
tic3 = time.time()
print("Normal estimation time (sec):", (tic3-tic2))
#o3d.visualization.draw_geometries([downpcd])


# In[126]:


#o3d.visualization.draw_geometries([downpcd],point_show_normal=True)


# In[226]:

'''
uni_down_pcd = pcd.uniform_down_sample(every_k_points=20)
uni_down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=100))
o3d.visualization.draw_geometries([uni_down_pcd],point_show_normal=True)
'''

# In[875]:


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (black): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])



# ## SOR

# In[228]:

'''
print("Statistical oulier removal")
cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=1.0)
display_inlier_outlier(uni_down_pcd, ind)
'''

# In[876]:


def  calculate_dipdirection(nx,ny):
    if nx>0:
        dip_direction = np.degrees(np.pi/2 - np.arctan(ny/nx))
    
    elif nx==0 and ny>=0:
        dip_direction = 0
        
    elif nx==0 and ny<0:
        dip_direction = 180
        
    else:
        dip_direction = np.degrees(np.pi*3/2 - np.arctan(ny/nx))
        
    return dip_direction
    

def  calculate_dip_delta(nx,ny,nz):
    if nx**2+ny**2==0:
        calculate_dip_delta = 0    
    else:
        calculate_dip_delta = np.degrees(np.pi/2 - np.arctan(abs(nz)/np.sqrt(nx**2+ny**2)))

    return calculate_dip_delta



normal_vector_2D=[]

for i in range(len(Normal_Vec_allpoints)):
    nx = Normal_Vec_allpoints[i][0]
    ny = Normal_Vec_allpoints[i][1]
    nz = Normal_Vec_allpoints[i][2]
    dip_direction = calculate_dipdirection(nx,ny)
    dip_delta = calculate_dip_delta(nx,ny,nz)
    normal_vector_2D.append([dip_direction,dip_delta])


normal_vector_2D=np.array(normal_vector_2D)
xyz_points = np.asarray(downpcd.points)
#clustering_type=['MEAN_SHIFT','AgglomerativeClustering','DBSCAN','OPTICS']

#X = StandardScaler().fit_transform(normal_vector_2D)
#X = sklearn.preprocessing.normalize(normal_vector_2D, axis=0, norm='max')

n_clusters_,labels = call_clustering('MEAN_SHIFT', Normal_Vec_allpoints)

tic4 = time.time()
print("Primary Clustering (sec):", (tic4-tic3))

# In[877]:


All = np.c_[xyz_points, labels]
 
cluster_0 = All[np.where(All[:,3]==0)[0]][:,0:3].astype('float32')
cluster_1 = All[np.where(All[:,3]==1)[0]][:,0:3].astype('float32')
cluster_2 = All[np.where(All[:,3]==2)[0]][:,0:3].astype('float32')
cluster_3 = All[np.where(All[:,3]==3)[0]][:,0:3].astype('float32')
cluster_4 = All[np.where(All[:,3]==4)[0]][:,0:3].astype('float32')
cluster_5 = All[np.where(All[:,3]==5)[0]][:,0:3].astype('float32')
cluster_6 = All[np.where(All[:,3]==6)[0]][:,0:3].astype('float32')
cluster_7 = All[np.where(All[:,3]==7)[0]][:,0:3].astype('float32')
cluster_8 = All[np.where(All[:,3]==8)[0]][:,0:3].astype('float32')

len(cluster_0)
   


# In[1045]:


### To automate each cluter with a number through list

Total_cluster = []
num_labels = len(list(set(labels)))

for i in range(num_labels):
    Total_cluster.extend([All[np.where(All[:,3] == i)[0]] [:, 0:3].astype('float32')])


len(Total_cluster)


# In[1051]:
ticc1 = time.time() 

for i in range(5):
    n_clusters_2,labels_2 = call_clustering('DBSCAN', Total_cluster[i])
    All_2 = np.c_[Total_cluster[i], labels_2]
    Total_cluster_2 = []
    num_labels_2 = len(list(set(labels_2)))
    for i in range(num_labels_2):
        Total_cluster_2.extend([All_2[np.where(All_2[:,3] == i)[0]] [:, 0:3].astype('float32')])
        pcd_sub = o3d.geometry.PointCloud()
        pcd_sub.points = o3d.utility.Vector3dVector(Total_cluster_2[i])
        pcd_np = np.asarray(pcd_sub.points)
        if len(np.asarray(pcd_sub.points)) > 3:
            plane_model, inliers = pcd_sub.segment_plane(distance_threshold=0.10,
                                         ransac_n=3,
                                         num_iterations=100)
            [a, b, c, d] = plane_model
            print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
            dip_dir = calculate_dipdirection(a,b)
            print("dip dir:", dip_dir)
            dip = calculate_dip_delta(a, b, c)
            print("dip :", dip)

ticc2 = time.time()
print("RANSAC (sec):", (ticc2-ticc1))
   
print("Total (sec):", (ticc2-tic2))     



