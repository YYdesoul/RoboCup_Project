import numpy as np
import cv2


class img_preprocessor:

    def __init__(self, paper_ratio=np.sqrt(2), pre_bilateral_filtering=False, post_bilateral_filtering=True,
                 bilateral_filter_distance=8, bilateral_filter_sigma_color=75,
                 bilateral_filter_sigma_space=75, threshold_block_size=89, threshold_c=4,
                 median_blur_size=11, approx_poly_epsilon_multiplier=0.1, rectangle_edge_length_epsilon_multiplier=0.1,
                 rectangle_aspect_epsilon_multiplier=0.1, rectangle_size_threshold=200,
                 cut_bilateral_filter_distance=5, cut_bilateral_filter_sigma_color=150,
                 cut_bilateral_filter_sigma_space=75, cut_threshold_block_size=89, cut_threshold_c=4, cut_offset=5,
                 debug=False):  # adjust the cut_bilateral_filter_distance to how thick the number should be in the cut
        """
        Initializes an img_preprocessor class. The parameters are calibration values for individual processing
        tasks. The relevant processing tasks and the parameters are described below.
        Bilateral filtering is a smoothing process, which uses neighboring pixels and their color. It calculates
        the median of the color values, but takes too big color changes into account to keep edges.
        Adaptive threshold filters each pixel and makes it white or black depending if it is above or below a threshold.
        The Threshold is calculates automatically in regard to the pixels around it.
        Median Blur smoothes the image again, after the threshold was applied.
        Shape detection is used to see if a contour matches a rectangle. The rectangles edges and aspect ration and
        size are checked to determine if the contour is a piece of paper.

        :param paper_ratio: The aspect ratio of the paper, which should be detected. The DIN norm uses root 2.
        :param pre_bilateral_filtering: Does a bilateral filtering before converting the image to gray scale.
            If there is little contrast, this might increase the results since it detects foreground better.
        :param post_bilateral_filtering: Does a bilateral filtering after converting the image to gray scale.
            Detects high contrast better, but merges fore- and background.
        :param bilateral_filter_distance: The pixel distance, which is considered. Higher values create better smoothing
        :param bilateral_filter_sigma_color: Color tolerance of pixels. Higher values create smoother objects
        :param bilateral_filter_sigma_space: Pixels are considered in this distance, if the color matches. Smoohtes edges
        :param threshold_block_size: Odd number greater 1. Higher values fuses objects together
        :param threshold_c: (can be negative) Subtracted from mean. Small positive numbers create more noise
        :param median_blur_size: Odd positive number. Higher values increase blurring
        :param approx_poly_epsilon_multiplier: Shape detection multiplier. Small values detect more edges
        :param rectangle_edge_length_epsilon_multiplier: 0<x<1; max difference of opposite edges in regards to their length
        :param rectangle_aspect_epsilon_multiplier: 0<x<1; max difference to proper aspect ratio in regards to their length
        :param rectangle_size_threshold: minimum paper size (in pixel)
        :param cut_bilateral_filter_distance: Same as above, but for the found paper
        :param cut_bilateral_filter_sigma_color: Same as above but for the found paper
        :param cut_bilateral_filter_sigma_space: Same as above but for the found paper
        :param cut_threshold_block_size: Same as above but for the found paper
        :param cut_threshold_c: Same as above but for the found paper
        :param cut_offset: An offset to the actually found image (in px), which is cut away to reduce error at the image edge
        :param debug: This will run otherwise unneeded code in order to show where the detected objects are located
        """
        self.paper_ratio = paper_ratio
        self.pre_bilateral_filtering = pre_bilateral_filtering
        self.post_bilateral_filtering = post_bilateral_filtering
        self.bilateral_filter_distance = bilateral_filter_distance
        self.bilateral_filter_sigma_color = bilateral_filter_sigma_color
        self.bilateral_filter_sigma_space = bilateral_filter_sigma_space
        self.threshold_block_size = threshold_block_size
        self.threshold_c = threshold_c
        self.median_blur_size = median_blur_size
        self.approx_poly_epsilon_multiplier = approx_poly_epsilon_multiplier
        self.rectangle_edge_length_epsilon_multiplier = rectangle_edge_length_epsilon_multiplier
        self.rectangle_aspect_epsilon_multiplier = rectangle_aspect_epsilon_multiplier
        self.rectangle_size_threshold = rectangle_size_threshold
        self.cut_bilateral_filter_distance = cut_bilateral_filter_distance
        self.cut_bilateral_filter_sigma_color = cut_bilateral_filter_sigma_color
        self.cut_bilateral_filter_sigma_space = cut_bilateral_filter_sigma_space
        self.cut_threshold_block_size = cut_threshold_block_size
        self.cut_threshold_c = cut_threshold_c
        self.cut_offset = cut_offset
        self.debug = debug

    def length(self, p1, p2):
        """Returns the length of the vector between two points"""
        return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def image_processing(self, img):
        """
        Processes the given image and detects all papers on it
        :param img: The image to process
        :return: The processed debug image, showing where papers are located and the papers.
            Each paper consists of:
                The image of the paper (a cut of the input image)
                A tuple of the four corner points (p1...p4)
                The center point
                A tuple of the edge length (length1...length4)
        """
        input_img = img
        border_padding = 5

        if self.pre_bilateral_filtering:
            # reduce noise, but keep edges. Smooth detection a lot
            img = cv2.bilateralFilter(img, self.bilateral_filter_distance,
                                      self.bilateral_filter_sigma_color, self.bilateral_filter_sigma_space)

        # convert to gray image TODO this might need to be changed, as the robots use a different color space
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.post_bilateral_filtering:
            # reduce noise, but keep edges. Smooth detection a lot
            img = cv2.bilateralFilter(img, self.bilateral_filter_distance,
                                      self.bilateral_filter_sigma_color, self.bilateral_filter_sigma_space)

        # converts each pixel to either black or white depending on color and surrounding pixels
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, self.threshold_block_size, self.threshold_c)

        # filter noise and smooth edges
        img = cv2.medianBlur(img, self.median_blur_size)

        # create a slight border in case paper touches the edge of the image
        img = cv2.copyMakeBorder(img, border_padding, border_padding, border_padding, border_padding,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # calculate the edges in image
        img = cv2.Canny(img, 200, 250)

        # find contours
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            # revert image to color for coloring debug
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # set up return values
        papers = []

        # Go through contours and check if a paper was found
        for c in contours:
            # find easier shapes with less polys
            epsilon = self.approx_poly_epsilon_multiplier * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:  # If easier contour has four points (hopefully it is a rectangle)
                # get points and lengths of edges
                p1, p2, p3, p4 = np.asarray(approx[0, 0]), np.asarray(approx[1, 0]), np.asarray(approx[2, 0]), \
                                 np.asarray(approx[3, 0])
                l1, l2, l3, l4 = self.length(p1, p2), self.length(p2, p3), self.length(p3, p4), self.length(p4, p1)
                l13_epsilon = self.rectangle_edge_length_epsilon_multiplier * ((l1 + l3) / 2)
                l24_epsilon = self.rectangle_edge_length_epsilon_multiplier * ((l2 + l4) / 2)

                # if opposite edges are about same length
                if np.abs(l1 - l3) < l13_epsilon and np.abs(l2 - l4) < l24_epsilon:

                    # Check how closely the object fits to the aspect of a paper
                    ratio_diff12 = l1 - (self.paper_ratio * l2) if l1 > l2 else l2 - (self.paper_ratio * l1)
                    ratio_diff34 = l3 - (self.paper_ratio * l4) if l3 > l4 else l4 - (self.paper_ratio * l3)
                    ratio_epsilon12 = ((l1 + (self.paper_ratio * l2) if l1 > l2 else l2 + (self.paper_ratio * l1)) / 2)\
                                      * self.rectangle_aspect_epsilon_multiplier
                    ratio_epsilon34 = ((l3 + (self.paper_ratio * l4) if l3 > l4 else l4 + (self.paper_ratio * l3)) / 2)\
                                      * self.rectangle_aspect_epsilon_multiplier
                    if np.abs(ratio_diff12) < ratio_epsilon12 and np.abs(ratio_diff34) < ratio_epsilon34:
                        #  Check if the area of the paper is the correct size
                        area = cv2.contourArea(approx)
                        if area > self.rectangle_size_threshold:
                            x, y, w, h = cv2.boundingRect(approx)  # get the bounding box

                            # Draw debug stuff to better show results
                            if self.debug:
                                cv2.drawContours(color_img, [approx], -1, (255, 0, 0), 3)  # Draw the contour
                                cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw the bounding box

                            # Cut the result from original image and process it
                            cut = np.asarray(input_img)[y - border_padding:y + h - border_padding,
                                          x - border_padding:x + w - border_padding]

                            if cut.size > 0:
                                cut = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
                                cut = cv2.bilateralFilter(cut, self.cut_bilateral_filter_distance,
                                                          self.cut_bilateral_filter_sigma_color,
                                                          self.cut_bilateral_filter_sigma_space)
                                cut = cv2.adaptiveThreshold(cut, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY_INV, self.cut_threshold_block_size,
                                                            self.cut_threshold_c)

                                # Calculate the center point of the paper
                                center = np.asarray([(p1[0] + p2[0] + p3[0] + p4[0]) / 4, (p1[1] + p2[1] + p3[1] + p4[1]) / 4])

                                # Calculate a slightly smaller shape. Everything around that shape is turned black
                                g1, g2, g3, g4 = center-p1, center-p2, center-p3, center-p4
                                g1, g2, g3, g4 = g1/np.linalg.norm(g1), g2/np.linalg.norm(g2), g3/np.linalg.norm(g3), \
                                                 g4/np.linalg.norm(g4)
                                g1, g2, g3, g4 = p1+(self.cut_offset*g1), p2+(self.cut_offset*g2), \
                                                 p3+(self.cut_offset*g3), p4+(self.cut_offset*g4)
                                p = np.array([x, y])
                                poly = np.asarray([[g1-p], [g2-p], [g3-p], [g4-p]])
                                poly = poly.astype(int)

                                border = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
                                cv2.fillPoly(cut, [poly, border], (0, 0, 0))

                                papers.append((cut, (p1, p2, p3, p4), center, (l1, l2, l3, l4)))

        # return (processed image, papers)
        if self.debug:
            return color_img, papers
        else:
            return img, papers


if __name__ == '__main__':
    processor = img_preprocessor(debug=True)  # create processor and turn on debug mode to see results

    # NOTE: The 0 + cv2.CAP_DSHOW might not work on every system. Try
    # vc = cv2.VideoCapture(0)
    # if it does not work.
    vc = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # Use the webcams image

    while(True):  # get the next frame and process it. Show the processed image
        ret, frame = vc.read()
        if (ret != True):
            print ("Webcam error")
            break
        proc, papers = processor.image_processing(frame)

        # show the processed image
        cv2.imshow('frame', proc)

        # if papers where found, show them as well
        if len(papers) > 0:
            cv2.imshow('papers', papers[0][0])
            # access the points like this
            #print papers[0][1] # tuple of the four points
            #print papers[0][2] # center points
            #print papers[0][3] # lengths of the edges

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()
