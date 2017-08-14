#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()
            # Define DH param symbols
            a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7') # link length
            d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8') # link offset
            alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7') # twist angles

            # Joint angle symbols
            q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8') # joint angles

            # Modified DH params
            DH_param_table = {alpha0:     0, a0:      0, d1:   0.75, q1:        q1,
                              alpha1: -pi/2, a1:   0.35, d2:      0, q2: -pi/2+ q2,
                              alpha2:     0, a2:   1.25, d3:      0, q3:        q3,
                              alpha3: -pi/2, a3: -0.054, d4:    1.5, q4:        q4,
                              alpha4:  pi/2, a4:      0, d5:      0, q5:        q5,
                              alpha5: -pi/2, a5:      0, d6:      0, q6:        q6,
                              alpha6:     0, a6:      0, d7:  0.303, q7:        0}
            
            # Define Modified DH Transformation matrix
            def Transform_Matrix(alpha, a, d, q):
                TF_M = Matrix([[ cos(q),    -sin(q),    0,  a],
                               [ sin(q)*cos(alpha),  cos(q)*sin(alpha),  -sin(alpha),    -sin(alpha)*d],
                               [ sin(q)*sin(alpha),  cos(q)*sin(alpha),  cos(alpha), cos(alpha)*d],
                               [ 0,  0,  0,  1]])
                return TF_M

            # Create individual transformation matrices
            T0_1 = Transform_Matrix(alpha0, a0, d1, q1).subs(DH_param_table)
            T1_2 = Transform_Matrix(alpha1, a1, d2, q2).subs(DH_param_table)
            T2_3 = Transform_Matrix(alpha2, a2, d3, q3).subs(DH_param_table)
            T3_4 = Transform_Matrix(alpha3, a3, d4, q4).subs(DH_param_table)
            T4_5 = Transform_Matrix(alpha4, a4, d5, q5).subs(DH_param_table)
            T5_6 = Transform_Matrix(alpha5, a5, d6, q6).subs(DH_param_table)
            T6_EE = Transform_Matrix(alpha6, a6, d7, q7).subs(DH_param_table)
            
            # Homogenous transformation matrix
            T0_EE = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_EE

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            r, p, y = symbols('r p y')

            # Roll
            R_x = Matrix([[ 1,  0,  0],
                          [ 0,  cos(r), -sin(r)],
                          [ 0,  sin(r), cos(r)]])

            # Pitch
            R_y = Matrix([[ cos(p), 0,  sin(p)],
                          [ 0,  1,  0],
                          [ -sin(p),    0,  cos(p)]])

            # Yaw
            R_z = Matrix([[ cos(y), -sin(y),    0],
                          [ sin(y), cos(y),    0],
                          [ 0,  0,  1]])

            R_EE = R_z * R_y * R_x 
            R_Error = R_z.subs(y, radians(180)) * R_y.subs(p, radians(-90))

            R_EE = R_EE * R_Error
            R_EE = R_EE.subs({'r': roll, 'p': pitch, 'y': yaw})

            EE_Matrix = Matrix([[px],
                                [py],
                                [pz]])

            WC = EE_Matrix - (0.303) * R_EE[:,2]

            # Calculate joint angles using Geometric IK method
            # Using Laws of Cosine
            theta1 = atan2(WC[1],WC[0])

            side_A = 1.501
            side_B = sqrt(pow((sqrt(WC[0]*WC[0] + WC[1]*WC[1]) - 0.35), 2) + pow((WC[2] - 0.75), 2))
            side_C = 1.25

            angle_a = acos((side_B*side_B + side_C*side_C - side_A*side_A)/(2*side_B*side_C))
            angle_b = acos((side_A*side_A + side_C*side_C - side_B*side_B)/(2*side_A*side_C))
            angle_c = acos((side_A*side_A + side_B*side_B - side_C*side_C)/(2*side_A*side_B))

            theta2 = pi/2 - angle_a - atan2(WC[2] - 0.75, sqrt(WC[0]*WC[0] + WC[1]*WC[1]) - 0.35)
            theta3 = pi/2 - (angle_b + 0.036)

            R0_3 = T0_1[0:3,0:3] * T1_2[0:3,0:3] * T2_3[0:3,0:3]
            R0_3 = R0_3.evalf(subs={q1: theta1, q2: theta2, q3: theta3})

            R3_6 = R0_3.T * R_EE 

            # Calculating joint angles for EE from rotation matrix
            theta4 = atan2(R3_6[2,2], -R3_6[0,2])
            theta5 = atan2(sqrt(R3_6[0,2]*R3_6[0,2] + R3_6[2,2]*R3_6[2,2]), R3_6[1,2])
            theta6 = atan2(-R3_6[1,1], R3_6[1,0])

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()
    
if __name__ == "__main__":
  IK_server()
