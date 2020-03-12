import numpy as np
import cv2
from img_preprocessor.img_preprocessor import img_preprocessor

from num_recognition.CNN_Net import Net
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


from location_calculator.location_calculator import location_calculator


class perception:

    def __init__(self):
        self.processor = img_preprocessor(debug=True)  # create processor and turn on debug mode to see results
        self.loc = location_calculator()

    def run(self, image, trans_matrix):
        proc, papers = self.processor.image_processing(image)

        cv2.circle(proc, (320, 240), 2, (0, 0, 255))

        # if papers where found, predict their number
        i = 0
        ret = []
        to_pil_image = transforms.ToPILImage()
        to_Tensor = transforms.ToTensor()
        for paper in papers:
            img = to_pil_image(paper[0]).resize((28, 28), Image.BILINEAR)
            img.save('paper_resize.png')
            img = to_Tensor(img)
            # print('img shape: ', img.shape)
            net = Net()
            net.load_state_dict(torch.load('num_recognition/checkpoints/test.pt'))
            out = net.forward(img.unsqueeze(0))
            _, predicted = torch.max(out.data, 1)
            #print('predict value for ' + str(i) + ': ', int(predicted[0]))
            #print(self.loc.calculate_position(paper))
            pos = self.loc.calculate_position(paper)
            pos = np.dot(trans_matrix, pos)

            ret.append((int(predicted[0]), pos))

            i = i + 1
        cv2.imshow('frame', proc)
        cv2.imwrite('frame.png', proc)
        if len(papers) > 0:
            cv2.imwrite('paper.png',papers[0][0])
            # cv2.imwrite('paper_resize.png',img)
        return proc, papers, ret


if __name__ == '__main__':
    # NOTE: The 0 + cv2.CAP_DSHOW might not work on every system. Try
    vc = cv2.VideoCapture(0)
    # if it does not work.
    #vc = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # Use the webcams image

    p = perception()
    trans_matrix = np.identity(3)

    while (True):  # get the next frame and process it. Show the processed image
        ret, frame = vc.read()
        if (ret != True):
            print("Webcam error")
            break

        proc, papers, predicted = p.run(frame, trans_matrix)

        # show the processed image
        cv2.imshow('frame', proc)
        i = 0
        for paper in papers:
            cv2.imshow('cut' + str(i), paper[0])
            i = i + 1

        i = 0
        for pred in predicted:
            print('predict value for ' + str(i) + ': ', pred[0], 'position: ', pred[1])
            i = i + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()