# Opencv helper functions and class
#import cv2
import numpy as np
from numpy import linalg as LA
#import pcl
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering, OPTICS, SpectralClustering, KMeans, MiniBatchKMeans, AffinityPropagation, Birch
#from sklearn.cluster import * 
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from sklearn.preprocessing import StandardScaler
#import pcl.pcl_visualization
import math

def convert_pointcloud_to_array(source):
  
    print(source.size)
    points = np.zeros((source.size, 4), dtype=np.float32)
    for i in range(0,source.size):
        points[i][0] = source[i][0]
        points[i][1] = source[i][1]
        points[i][2] = source[i][2]
        points[i][3] = source[i][3]   
        print('counter:',i)
    return points


def merge(points1, points2):
   
    """
    merge two points cloud data
    """
    N = points1.shape[0]
    M = points2.shape[0]
    final_points= np.zeros((N+M, 4), dtype=np.float32)
    #print(N)
    #print(M)
    
    for i in range(0,N): 
          final_points[i][0]=points1[i][0]
          final_points[i][1]=points1[i][1]
          final_points[i][2]=points1[i][2]
          final_points[i][3]=points1[i][3]
    for i in range(1,M):
          final_points[i+N][0]=points2[i][0]
          final_points[i+N][1]=points2[i][1]
          final_points[i+N][2]=points2[i][2]
          final_points[i+N][3]=points2[i][3]
    return final_points

def remove_plan(points):
    
    # Ransac 
    cloud_3 = pcl.PointCloud()
    cloud_3.from_array(points)
    m = points.shape[0]
    #Planar object
    seg = cloud_3.make_segmenter_normals(ksearch=200)
    LX=cloud_3.make_segmenter_normals(ksearch=200)
    print(LX)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    #seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.2) # (0.09)
    #seg.set_normal_distance_weight(0.1)
    seg.set_max_iterations(1000)
    indices, coefficients = seg.segment() # indices are the index of inlier points in Ransac
    print(coefficients)
    
    n=len(indices)
    print('The length of indices:', n)
    points12= np.zeros((m-n, 3), dtype=np.float32)
    points13= np.zeros((n, 4), dtype=np.float32)
    a=0
    c=0
    
    for k in range(0,m):
        b=0
        for indexx in indices:
            if (k==indexx):
                b=1
                points13[c][0]=points[k][0]
                points13[c][1]=points[k][1]
                points13[c][2]=points[k][2]
                #points13[c][3]=255 <<16|255<<8|255
                points13[c][3]=255 |0|0
                c=c+1
            
        if (b==0):             
           points12[a][0]=points[k][0]
           points12[a][1]=points[k][1]
           points12[a][2]=points[k][2]
           a=a+1
    return points12,points13

# VG_size = 0.04
def cluster_DBSCAN(final_points):
    # dbscan 
    clustering = DBSCAN(eps = 0.1, min_samples=10, leaf_size=30).fit(final_points)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_# Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    n_clusters_ = len(set(labels))
    n_noise_ = list(labels).count(-1)
    print('DBSCAN')
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Number of clusters in labels:',set(labels))
    return n_clusters_,labels
  
    
def cluster_mean_shift(final_points):
    bandwidth = estimate_bandwidth(final_points, quantile=0.2, n_samples=50)

    ms = MeanShift(bandwidth = 0.28, bin_seeding=True)
    ms.fit(final_points)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('MeanShift')
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Number of clusters in labels:',set(labels))
    return n_clusters_,labels

def cluster_AgglomerativeClustering(final_points):
    clustering = AgglomerativeClustering().fit(final_points)
    labels=clustering.labels_ 
    n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('AgglomerativeClustering')
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Number of clusters in labels:', set(labels))
    return n_clusters_,labels

def Cluster_OPTICS(final_points):
    clustering = OPTICS(min_samples=2).fit(final_points)
    labels=clustering.labels_
    n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Number of clusters in labels:', set(labels))
    return n_clusters_,labels


## Gaussian Mixture Models ()
    
def Gaussian_Mixture(final_points):
    gmm = GaussianMixture(n_components = 5, covariance_type = 'full', random_state = 0)
    gmm.fit(final_points)
    labels = gmm.predict(final_points)
    n_clusters_ = len(set(labels))
    return n_clusters_,labels


# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
# Bayesian Gaussian Mixture
# Fit a Dirichlet process Gaussian mixture using five components
    
def bayesian_GM(final_points):
    dpgmm = BayesianGaussianMixture(n_components = 5, covariance_type="full", random_state = 2)
    dpgmm.fit(final_points)
    labels = dpgmm.predict(final_points)
    n_clusters_ = len(set(labels))
    return n_clusters_,labels


def Spectral_Clustering(final_points):
    clustering = SpectralClustering(n_clusters=5, eigen_solver = 'amg', affinity ='rbf', assign_labels='discretize', n_jobs=-1).fit(final_points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels))
    return n_clusters_,labels


def kmeans(final_points):
    clustering = KMeans(n_clusters = 5, random_state=0).fit(final_points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels))
    return n_clusters_,labels


def MBKmeans(final_points):
    clustering = MiniBatchKMeans(n_clusters = 5,
                          random_state = 0,
                          batch_size = 6,
                          max_iter = 10).fit(final_points)
    
    labels = clustering.labels_
    n_clusters_ = len(set(labels))
    return n_clusters_,labels


# https://scikit-learn.org/dev/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation

def affinity(final_points):
    clustering = AffinityPropagation().fit(final_points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels))
    return n_clusters_,labels

def birch(final_points):
    clustering = Birch(n_clusters=None).fit(final_points)
    labels = clustering.labels_
    n_clusters_ = len(set(labels))
    return n_clusters_,labels
    
    





# ===========================================================


def normals(source):
    global Q
    cloud = pcl.PointCloud()
    m=source.shape[0]
    points= np.zeros((m, 3), dtype=np.float32)
    for i in range(0,m):
        points[i][0] = source[i][0]
        points[i][1] = source[i][1]
        points[i][2] = source[i][2]
    
    print(type)
    U, s, V = np.linalg.svd(points)
    print(V)
    N = np.zeros((3, 1), dtype=np.float32)
    N1 = -1/V[2][2]
    N[0] = N1*V[0][2]
    N[1] = N1*V[1][2]
    N[2] = N1*V[2][2]
    A = N[0]
    B = N[2]

    print(V)
    N = np.zeros((3, 1), dtype=np.float32)
    N1 = -1/V[2][2]
    N[0] = N1*V[0][2]
    N[1] = N1*V[1][2]
    N[2] = N1*V[2][2]
    A = N[0]
    B = N[2]
    C = -(N[0] * N[0] + N[1] * N[1] + N[2] * N[2])
    
    theta = np.arctan(B / A) * 180 / np.pi
    delta = np.arctan((np.sqrt(C*C+B*B))) * 180 / np.pi
    if (A > 0 and m < 0):
       Q = 360
    if (A > 0 and m > 0):
       Q = 0 
    if (A < 0 and m < 0):
       Q = 180  
    if (A < 0 and m > 0):
       Q = 180
    
    theta = theta + Q         
    print('theta= '+str(theta))
    print('delta= '+str(delta))


def Finding_normal(source):
    global Q
    cloud = pcl.PointCloud()
    m=source.shape[0]
    points= np.zeros((m, 3), dtype=np.float32)
    for i in range(0,m):
        points[i][0] = source[i][0]
        points[i][1] = source[i][1]
        points[i][2] = source[i][2]
    

    U, s, V = np.linalg.svd(points)   
    min_index = np.argmin(s,axis=0)   # eigen value
    normal_vector = V[:,min_index]
    
   
    return  normal_vector





def call_clustering(name, final_points):
    if name == 'DBSCAN':
        n_clusters_,labels = cluster_DBSCAN(final_points)        
    elif name == 'MEAN_SHIFT':
        n_clusters_,labels = cluster_mean_shift(final_points) 
    elif name == 'AgglomerativeClustering':
        n_clusters_,labels = cluster_AgglomerativeClustering(final_points) 
    elif name == 'OPTICS':
        n_clusters_,labels = Cluster_OPTICS(final_points)
    elif name == 'GMix':
        n_clusters_,labels = Gaussian_Mixture(final_points)
    elif name == 'BGMix':
        n_clusters_,labels = bayesian_GM(final_points)
    elif name == 'Spectral':
        n_clusters_,labels = Spectral_Clustering(final_points)
    elif name == 'kmeans':
        n_clusters_,labels = kmeans(final_points)
    elif name == 'MBKmeans':
        n_clusters_,labels = MBKmeans(final_points)
    elif name == 'affinity':
        n_clusters_,labels = affinity(final_points)
    elif name == 'birch':
        n_clusters_,labels = birch(final_points)
       
    else:        
        print('there is no error for calling')
        n_clusters_,labels = cluster_mean_shift(final_points)
        
    return n_clusters_,labels
    

def convert_open3dformat_to_pclformat(dataopen3d):
    
    array_open3d=np.asarray(pcd.points)
    pointcloud_instance= pcl.PointCloud()
    array_open3d=array_open3d.astype(np.float32)
    PCL_type_data=pointcloud_instance.from_array(array_open3d)
    PCL_type_data=pointcloud_instance.from_array(array_open3d)
    return array_open3d    


def compute_curvature(pcd, radius=0.5):

    points = np.asarray(pcd.points)

    from scipy.spatial import KDTree
    tree = KDTree(points)

    curvature = [ 0 ] * points.shape[0]

    for index, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)

        # local covariance
        M = np.array([ points[i] for i in indices ]).T
        M = np.cov(M)

        # eigen decomposition
        V, E = np.linalg.eig(M)
        # h3 < h2 < h1
        h1, h2, h3 = V

        curvature[index] = h3 / (h1 + h2 + h3)

    return curvature