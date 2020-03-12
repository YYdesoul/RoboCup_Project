from naoqi import ALProxy
import time
import numpy as np
import math
import vision_definitions
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import sys
from PyQt4.QtGui import QWidget, QImage, QApplication, QPainter
import PyQt4
from location_calculator import location_calculator
from img_preprocessor import img_preprocessor
from run_perception import perception
from TestCNN import Net
from camera_test_video import ImageWidget

class Motion():

    def __init__(self):
        self.alive = True
        self.robot_ip = "nao34.local"
        self.port = 9559
        self.ttsProxy = ALProxy("ALTextToSpeech", self.robot_ip,self.port)
        self.motionProxy = ALProxy("ALMotion", self.robot_ip, self.port)
        self.postureProxy = ALProxy("ALRobotPosture", self.robot_ip, self.port)
        self.camProxy = ALProxy("ALVideoDevice", self.robot_ip, self.port)
        self.app = QApplication(sys.argv)
        self.imageWidget = ImageWidget(self.robot_ip, self.port, 0)
        self.motionProxy.rest()
        self.perception = perception()



    def default_position(self):
        self.motionProxy.angleInterpolationWithSpeed(['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                                                     [1.3974320888519287, 0.15335798263549805, -0.7777800559997559, -1.0400099754333496], 0.5)
        self.motionProxy.angleInterpolationWithSpeed(
            ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
            [1.4082541465759277, -0.14883995056152344, 0.7822980880737305, 1.0370259284973145], 0.5)
        self.motionProxy.setStiffnesses("Body", 0.0)

    def point_to_number(self, x, y, z):
        if x == 0 and y == 0 and z == 0:
            return;
        else:
            if y >= 0:
                anglePitch = (-1) * np.arctan((z+90.0)/x)
                angleRoll = np.arctan((y-98.0)/x) - np.arctan(15.0 / 105.0)
                if angleRoll < -0.3142:
                    angleRoll = -0.3042
                self.motionProxy.setStiffnesses("Body", 1.0)
                names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
                angles = [anglePitch, angleRoll - np.arctan(15. / 105.), 0, 0]
                fractionMaxSpeed = 0.7
                self.motionProxy.angleInterpolationWithSpeed(names,angles,fractionMaxSpeed)
            else:
                anglePitch = (-1) * np.arctan((z+90.0) / x)
                angleRoll = np.arctan((y + 98.0) / x) + np.arctan(15.0 / 105.0)
                if angleRoll > 0.3142:
                    angleRoll = 0.3042
                self.motionProxy.setStiffnesses("Body", 1.0)
                names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
                angles = [anglePitch, angleRoll - np.arctan(15. / 105.), 0, 0]
                fractionMaxSpeed = 0.7
                self.motionProxy.angleInterpolationWithSpeed(names, angles, fractionMaxSpeed)

    def get_angles(self):
        names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
        useSensors = False
        commandAngles = self.motionProxy.getAngles(names, useSensors)
        print "Command angles:"
        print str(commandAngles)
        print ""

        useSensors = True
        sensorAngles = self.motionProxy.getAngles(names, useSensors)
        print "Sensor angles:"
        print str(sensorAngles)
        print ""

    #TODO
    def get_position(self,ret):


        if len(ret) > 0:
            print ret[0][1]
            return ret[0][1]
        else:
            return (0,0,0)

    def image(self):

        resolution = 2  # VGA
        colorSpace = 11  # RGB

        videoClient = self.camProxy.subscribe("python_client", resolution, colorSpace, 5)

        naoImage = self.camProxy.getImageRemote(videoClient)

        self.camProxy.unsubscribe(videoClient)

        # Now we work with the image returned and save it as a PNG  using ImageDraw
        # package.

        # Get the image size and pixel array.
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]

        # Create a PIL Image from our pixel array.
        im = Image.frombytes("RGB", (imageWidth, imageHeight), array)
        im.show()
        pic = np.array(im)
        return pic

    def get_number(self,ret):
        if len(ret) > 0:
            return ret[0][0]
        else:
            return -1

    def speak_number(self,ret):
        number = self.get_number(ret)
        if number < 0:
            self.ttsProxy.say("no number")
        else:
            self.ttsProxy.say("the number is " + str(number))

    def motion_loop(self):
        self.default_position()
        matr = np.identity(3)
        for i in range(10):
            image, _, ret = self.perception.run(self.image(), matr)
            x,y,z = self.get_position(ret)
            self.point_to_number(x, y, z)
            self.speak_number(ret)
            self.default_position()


    def aa(self):
        self.motionProxy.setStiffnesses("Body", 1.0)
        self.motionProxy.angleInterpolationWithSpeed(['LShoulderPitch','LShoulderRoll', 'LElbowYaw', 'LElbowRoll'], [-0.5,0,0,0], 0.5)
        self.default_position()



if __name__ == '__main__':

    agent = Motion()
    agent.motion_loop()
