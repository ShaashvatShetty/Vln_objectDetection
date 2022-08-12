import drone_simulator
import numpy as np
import cv2
import torch
import unet
#import detect_testing
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from drone_simulator import Drone
#from detect_testing import get_object_coord
# from simulator_modified import DroneSimulator
#from detect_testing import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['Water', 'Land', 'Rocks', 'Road', 'Grass', 'Vegetation', 'Tree',
           'Building', 'Window', 'Fence', 'Car', 'Bicycle', 'Person']


def get_mask(image, model_path):
    model = unet(n_channels=3, n_classes=28, bilinear=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    image = cv2.resize(image, (512, 512)) / 255.0
    image = np.moveaxis(image, -1, 0)
    image = torch.tensor(image).float()
    image = torch.unsqueeze(image, dim=0)
    result = model(image.to(device))
    mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]

    return mask

    # image = np.moveaxis(image.to(device)[0].cpu().detach().numpy(), 0, -1).copy() * 255
    # image = image.astype(int)

    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap='gray_r')

def get_closest_loc(pos, loc_list):
    x = pos[0]
    y = pos[1]
    min_dist = np.inf
    close_loc = pos
    for loc in loc_list:
        new_x = loc[0]
        new_y = loc[1]
        dist = np.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
        if dist < min_dist:
            min_dist = dist
            close_loc = loc
    return close_loc

class DroneSimulator:
    def __init__(self, classes: list, img_path: str):
        name_idx = img_path.rindex('/')
        img_name = img_path[name_idx + 1:]

        self.drone = Drone(img_name)
        self.drone.move(self.drone.getXCoordinate(), self.drone.getYCoordinate())
        self.classes = classes
        self.image = cv2.imread(img_path)

    def find_closest_obj(self, obj: str, pos: list, direction='f'):
        """
        Returns the coordinates of the obj closest to the drone.
        """
        mask = get_mask(self.image, f'C:/Users/pranav/Desktop/SIP2022/UNet_Aerial_Segmentation/best_models/{obj}.pt')
        plt.imshow(mask, cmap='gray')
        x_pos = pos[0]
        y_pos = pos[1]
        plt.plot([x_pos], [y_pos], '^', markersize=15, color='yellow', markeredgecolor='red')

        loc_list = []

        if direction == 'f':
            for r_idx in range(len(mask)):
                for c_idx in range(y_pos, 0, -1):
                    if mask[r_idx][c_idx] == 27:
                        loc_list.append([r_idx, c_idx])

            new_pos = get_closest_loc(pos, loc_list)
            plt.plot([new_pos[0]], [new_pos[1]], '^', markersize=15, color='yellow', markeredgecolor='red')
            plt.show()
            return new_pos

    # def move_to(self, new_pos):
    #     self.drone.move(ord('2'))

if __name__ == '__main__':
    img_path = '/579.tif'
    drone = DroneSimulator(classes, img_path)
    # drone.find_closest_obj('Building', [200, 200])
    # drone.move_to([2, 4])