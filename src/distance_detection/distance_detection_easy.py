
ref_l1 = 141,9  # length in pixel
ref_l2 = 103,85 # width in pixel
ref_distance = 1000  # millimeter

def distance_detection(l1, l2, l3, l4):
    """
    Prameters: l1, l2, l3, l4 - Lengths of detected rectangle
    Returns: Distance in millimeteres from the recangle to the camera
    """
    avg_l1_l3 = (l1+l3)/2
    avg_l2_l4 = (l2+l4)/2

    input_area = avg_l1_l3 * avg_l2_l4 # area for cross multiplication
    ref_area = ref_l1 * ref_l2 # reference area for cross multiplication
    # cross multiplication: ref_distance / ref_area = distance / input_area
    distance = (ref_distance * input_area) / ref_area # cross multiplication transformed 

    return distance