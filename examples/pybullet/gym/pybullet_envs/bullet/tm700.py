import pybullet as p
import time
import pybullet_data
import os, inspect
import numpy as np
import copy
import math


class tm700:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1
    self.tmEndEffectorIndex = 7
    self.tmGripperIndex = 10
    # lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    # upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    # joint ranges for null space
    # self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    self.jr = [10, 10, 10, 10, 10, 10, 10]
    # restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.reset()

  def reset(self):

    robot = p.loadURDF("../Gazebo_arm/urdf/tm700_robot.urdf")
    self.tm700Uid = robot
    p.resetBasePositionAndOrientation(self.tm700Uid, [0.15, 0.0, 0.0], # position of robot, GREEN IS Y AXIS
                                      [0.000000, 0.000000, 1.000000, 0.000000]) # direction of robot
    self.jointPositions = [
        0.0, 0.0, -0, -0, -0.5, -1, -1.57, 0,
        -0, -0, -0, -0, -0, -0.0, -0.0, -0.0
    ]

    self.numJoints = p.getNumJoints(self.tm700Uid)
    for jointIndex in range(self.numJoints):
        p.resetJointState(self.tm700Uid, jointIndex, self.jointPositions[jointIndex])
        p.setJointMotorControl2(self.tm700Uid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

        # print('Link:', p.getLinkState(self.tm700Uid, jointIndex))

        # print(p.getJointInfo(robot, jointIndex))



    self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.6400, #first 3: position, last 4: quaternions
                              0.0000, 0.001, 0.000000, 0.000000, 1.000000, 0.000000)
    self.endEffectorPos = [0.537, 0.0, 0.5]
    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.tm700Uid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)
        # print('motorindeces', self.motorIndices)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

    jointInfo = p.getJointInfo(self.tm700Uid, i)
    qIndex = jointInfo[3]

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.tm700Uid, self.tmGripperIndex)
    pos = state[0]
    orn = state[1] #Cartesian orientation of center of mass, in quaternion [x,y,z,w]

    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation

  def applyAction(self, motorCommands):

    #print ("self.numJoints")
    #print (self.numJoints)
    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]
      state = p.getLinkState(self.tm700Uid, self.tmEndEffectorIndex) # returns 1. center of mass cartesian coordinates, 2. rotation around center of mass in quaternion
      actualEndEffectorPos = state[0]
      #print("pos[2] (getLinkState(tmEndEffectorIndex)")
      #print(actualEndEffectorPos[2])

      self.endEffectorPos[0] = self.endEffectorPos[0] + dx
      if (self.endEffectorPos[0] > 0.65):
        self.endEffectorPos[0] = 0.65
      if (self.endEffectorPos[0] < 0.50):
        self.endEffectorPos[0] = 0.50
      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      if (self.endEffectorPos[1] < -0.17):
        self.endEffectorPos[1] = -0.17
      if (self.endEffectorPos[1] > 0.22):
        self.endEffectorPos[1] = 0.22

      #print ("self.endEffectorPos[2]")
      #print (self.endEffectorPos[2])
      #print("actualEndEffectorPos[2]")
      #print(actualEndEffectorPos[2])
      #if (dz<0 or actualEndEffectorPos[2]<0.5):
      self.endEffectorPos[2] = self.endEffectorPos[2] + dz
  #
      self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.tm700Uid, self.tmEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.tm700Uid,
                                                    self.tmEndEffectorIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.tm700Uid,
                                                    self.tmEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.tm700Uid, self.tmEndEffectorIndex, pos)


      #print("self.tmEndEffectorIndex")
      #print(self.tmEndEffectorIndex)
      if (self.useSimulation):
        for i in range(self.tmEndEffectorIndex):

          p.setJointMotorControl2(bodyUniqueId=self.tm700Uid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.tm700Uid, i, jointPoses[i])
      #fingers
      # p.setJointMotorControl2(self.tm700Uid,
      #                         7,
      #                         p.POSITION_CONTROL,
      #                         targetPosition=self.endEffectorAngle,
      #                         force=self.maxForce)

      p.setJointMotorControl2(self.tm700Uid,
                              11,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
      p.setJointMotorControl2(self.tm700Uid,
                              12,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)

    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.tm700Uid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)


# if __name__ == '__main__':


    # physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    #
    # tm700test = tm700()
    # tm700test.reset
    # tm700test.applyAction([0.67, 0.2, 0.01,-1,5])
    # for i in range (10000):
    #     p.stepSimulation()
    #     time.sleep(1./240.0)
    #
    # p.disconnect()
