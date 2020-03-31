import numpy as np
# import matplotlib.pyplot as plt

class location_calculator:

    def __init__(self, camera_angle_horizontal=60.97, camera_angle_vertical=47.64, screen_width=640, screen_height=480,
                 paper_width=210, paper_height=297):
        self.camera_angle_horizontal = camera_angle_horizontal
        self.camera_angle_vertical = camera_angle_vertical
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paper_width = paper_width
        self.paper_height = paper_height

    def calculate_distance(self, camera_angle, paper_length_px, paper_length_mm, screen_px, paper_position):
        """
        calculates the distance of a paper
        :param paper_length_px: The papers length in pixel (on an image)
        :param paper_length_mm: The actual length of the paper in the real world in mm
        :param screen_px: the length of the screen in px (can be either width or height depending on papers orientation)
        :param paper_position: paper position on the image in px
        :return: The distance on the paper in mm
        """
        # calculate length of screen in mm
        a = (paper_length_mm * screen_px) / paper_length_px

        # calculate angles of triangle (only alpha and gamma are needed and also beta = gamma)
        alpha = np.radians(camera_angle)
        gamma = (np.pi - alpha) / 2

        # calculate length of c
        c = a * np.sin(gamma) / np.sin(alpha)

        # calculate the cut at which the actual distance cuts the angle alpha
        rel_alpha = paper_position / screen_px * alpha

        # calculate new angle delta using the new alpha and gamma
        delta = np.pi - rel_alpha - gamma

        # calculate the actual distance d
        d = c * np.sin(delta) / np.sin(gamma)

        return d

    def calculate_angular_distance(self, distance, camera_angle, screen_size, paper_position):
        """
        calculates the distance from the center point on one axis
        :param distance: the distance of the paper
        :param camera_angle: the camera angle
        :param screen_size: the screen size (either width or height depending on orientation)
        :param paper_position: the position of the paper on the image in pixel
        :return: the distance to the center position
        """
        cam_angle = np.radians(camera_angle)
        angle = (paper_position / screen_size * cam_angle) - (cam_angle / 2)
        # print("distance:" , distance)
        # print "camera angle :" ,camera_angle
        # print "screen size :" , screen_size
        # print "paper position:" , paper_position
        # print "angle" ,np.degrees(angle)
        return np.sin(angle) * distance

    def calculate_position(self, paper):
        """
        calculates the position of a paper in the real world in reference to the cameras position
        :param paper: the paper as (processed image, (tuple of corner points), center point, (tuple of lengths)).
            Only the center point and lengths are needed.
        :return: a vector to the center of the paper in reference to the cameras position
        """
        lengths = np.array(paper[3])
        lengths.sort()
        width = float((lengths[0] + lengths[1]) / 2)
        height = float((lengths[2] + lengths[3]) / 2)

        dist1 = self.calculate_distance(float(self.camera_angle_horizontal), float(width), float(self.paper_width), float(self.screen_width), float(paper[2][0]))
        dist2 = self.calculate_distance(float(self.camera_angle_vertical), float(height), float(self.paper_height), float(self.screen_height), float(paper[2][1]))

        dist = (dist1 + dist2) / 2

        #print(dist1, dist2, dist)
        # print("center: ",paper[2])
        pos_w = -self.calculate_angular_distance(dist, float(self.camera_angle_horizontal), float(self.screen_width), paper[2][0])
        pos_h = -self.calculate_angular_distance(dist, float(self.camera_angle_vertical), float(self.screen_height), paper[2][1])
        # print(pos_w,pos_h)
        return np.array([dist, pos_w, pos_h])


if __name__ == '__main__':
    calc = location_calculator()
    paper = (None, ([], [], [], []), [505, 311], (89.9, 135.2, 96.8, 135.7))
    print(calc.calculate_position(paper))
