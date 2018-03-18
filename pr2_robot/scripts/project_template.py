#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import pcl


def statistical_outlier_filter(cloud_filtered):
    # Much like the previous filters, we start by creating a filter object:
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)
    # Set threshold scale factor
    x = 0.1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()
    return cloud_filtered


def vox_filter(cloud):
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
    return cloud_filtered


def passthrough_filter(cloud_filtered, axis, min, max):
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = axis
    passthrough.set_filter_field_name(filter_axis)
    axis_min = min
    axis_max = max
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    return cloud_filtered


def RANSAC_filter(cloud_filtered):
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    return inliers


def get_white_cloud(extracted_outliers):
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()
    return tree, white_cloud


def euclidian_cluster(method, white_cloud):
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(20000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(method)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    return cluster_indices

def cluster_coloring(cluster_indices, white_cloud):
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def publish_cloud(pub, cloud):
    ros_cloud_filtered = pcl_to_ros(cloud)
    pub.publish(ros_cloud_filtered)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    cloud = ros_to_pcl(pcl_msg)

    cloud_filtered = statistical_outlier_filter(cloud)
    cloud_filtered = vox_filter(cloud_filtered)
    cloud_filtered = passthrough_filter(cloud_filtered, axis='z', min=0.6, max=1.1)
    cloud_filtered = passthrough_filter(cloud_filtered, axis='y', min=-0.5, max=0.5)
    inliers = RANSAC_filter(cloud_filtered)

    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    tree, white_cloud = get_white_cloud(extracted_outliers)

    cluster_indices = euclidian_cluster(tree, white_cloud)

    # Assign a color corresponding to each segmented object in scene
    cluster_cloud = cluster_coloring(cluster_indices, white_cloud)

    publish_cloud(pcl_objects_pub, extracted_outliers)
    publish_cloud(pcl_table_pub, extracted_inliers)
    publish_cloud(pcl_cluster_pub, cluster_cloud)

    detected_objects_list = detect_objects(cluster_indices, extracted_outliers, white_cloud)

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects_list)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass


def detect_objects(cluster_indices, extracted_outliers, white_cloud):
    detected_objects_labels = []
    detected_objects_list = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects_list.append(do)
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    return detected_objects_list


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    scene = 2
    centroids = []  # to be list of tuples (x, y, z)
    objects_map = {}
    dropbox_map = {}

    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    for object_param in object_list_param:
        objects_map[object_param['name']] = object_param['group']

    for dropbox in dropbox_param:
        dropbox_map[dropbox['group']] = (dropbox['name'], dropbox['position'])


    dict_list = []

    for object in object_list:
        if object.label not in objects_map:
            rospy.loginfo('Request object not found %s' % object.label)
            print(object.label)
            print(objects_map)
            continue

        group = objects_map[object.label]
        dropbox_name, dropbox_pos = dropbox_map[group]

        points_arr = ros_to_pcl(object.cloud).to_array()
        centoroid = np.mean(points_arr, axis=0)[:3]
        centroids.append(centoroid)

        PLACE_POSE = Pose()
        PLACE_POSE.position.x = float(dropbox_pos[0])
        PLACE_POSE.position.y = float(dropbox_pos[1])
        PLACE_POSE.position.z = float(dropbox_pos[2])

        PICK_POSE = Pose()
        PICK_POSE.position.x = float(centoroid[0])
        PICK_POSE.position.y = float(centoroid[1])
        PICK_POSE.position.z = float(centoroid[2])

        WHICH_ARM = String()
        WHICH_ARM.data = dropbox_name

        TEST_SCENE_NUM = Int32()
        TEST_SCENE_NUM.data = scene
        OBJECT_NAME = String()
        OBJECT_NAME.data = str(object.label)

        yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM, OBJECT_NAME, PICK_POSE, PLACE_POSE)
        dict_list.append(yaml_dict)

        '''
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')
        
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        '''
    # TODO: Output your request parameters into output yaml file
    send_to_yaml('/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/output_{}.yaml'.format(scene), dict_list)


if __name__ == '__main__':
    rospy.init_node('clustering', anonymous=True)
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    while not rospy.is_shutdown():
        rospy.spin()
