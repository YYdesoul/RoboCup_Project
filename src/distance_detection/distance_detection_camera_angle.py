import numpy as np 

"""
Preamble:
We assume the paper is in a 90 degree angle to our distance:


          /|
       b / |
        /  |
       /   |
      /   ||
ROBOT ────|| a
      \   ||
       \   |
       c\  |
         \ |
          \|
Goal is to calculate b and c       
"""

def distance_detection(alpha, px_of_pic=50, paper_width_in_pic):

    a_in_px             = pixel_of_pic
    paper_width_in_pic  = paper_width_in_pic
    paperwidth          = 210 # fix width of a paper 210mm

    #crossmultiplication to calculate a:
    # a / pixel_of_pic  = paperwidth / paper_width_in_pic <=>
    a                   = (paperwidth / paper_width_in_pic) * pixel_of_pic

    #calculate beta and gamme due to inside angle sum
    beta, gamma         = (180 - alpha)/2

    #calculate b and c with Law of sines:
    #since beta = gamma we can say b = c
    # a / sin(alpha) = b / sin(beta) = c / sin(gamma) <=> 
    b, c                = (a / np.sin(alpha)) * np.sin(beta)

    #return the length of a hypotenuse
    return b
