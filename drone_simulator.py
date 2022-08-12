import cv2
import numpy as np
import pickle
import scipy.spatial as spt
import detect_testing
from detect_testing import *

from detect import detect


parser = argparse.ArgumentParser()
# Get data configuration

parser.add_argument('-image_folder', type=str, default='/drone_images/579.tif')
parser.add_argument('-output_folder', type=str, default='./output_xview', help='path to outputs')
cuda = False  # torch.cuda.is_available()

parser.add_argument('-plot_flag', type=bool, default=True)
parser.add_argument('-secondary_classifier', type=bool, default=False)
parser.add_argument('-cfg', type=str, default='/xview-yolov3/cfg/c60_a30symmetric.cfg', help='cfg file path')
parser.add_argument('-class_path', type=str, default='/xview-yolov3/data/xview.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.99, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=32 * 51, help='size of each image dimension')
opt = parser.parse_args()

#x = detect(opt, 1, "car","nearest",0,0)
#print(x)

# from detect_testing import getY
# you will need to download:
# 18 folder
# .csv (tells you to use 18.tif, automatic)
# /Users/fanyue/xview/xView_train.geojson

# input files:
# /Users/fanyue/xview/18/{*.pickle}
# /Users/fanyue/xview/train_images/18.tif
# /Users/fanyue/Downloads/Batch_4582824_batch_results.csv

# output:
# /Users/fanyue/xview/18/{*_1.pickle}
# /Users/fanyue/xview/18/{image_sample_*_1.jpg}

# steps:
# wasd+oi control, press 'esc'
# input: x means reject. y means already found it. otherwise, questions.
# press enter

# pre_rejected_list_df = pd.read_csv('/Users/fanyue/Downloads/Batch_4632300_batch_results.csv_second_filtered.csv')


def polygon_area(points):
    hull = spt.ConvexHull(points=points)
    return hull.area


def get_a_gps_coord_at_distance(a, b):
    # a is a gps coord
    # b is a distance in meter
    # return a gps coord
    return b / 11.13 / 1e4 + a


def rotation_anticlock(theta, p):
    M = np.array([[np.cos(theta / 180 * 3.14159), -np.sin(theta / 180 * 3.14159)],
                  [np.sin(theta / 180 * 3.14159), np.cos(theta / 180 * 3.14159)]])
    return np.matmul(M, np.array([p[0], p[1]]))


def change_corner(cs, change):  # corners = cs
    new_cs = np.zeros((4, 2))
    new_cs[0] = cs[0] + (cs[1] - cs[0]) / np.linalg.norm((cs[1] - cs[0])) * change[0][0]
    new_cs[0] += (cs[3] - cs[0]) / np.linalg.norm((cs[3] - cs[0])) * change[0][1]

    new_cs[1] = cs[1] + (cs[1] - cs[0]) / np.linalg.norm((cs[1] - cs[0])) * change[1][0]
    new_cs[1] += (cs[2] - cs[1]) / np.linalg.norm((cs[2] - cs[1])) * change[1][1]

    new_cs[2] = cs[2] + (cs[2] - cs[3]) / np.linalg.norm((cs[2] - cs[3])) * change[2][0]
    new_cs[2] += (cs[2] - cs[1]) / np.linalg.norm((cs[2] - cs[1])) * change[2][1]

    new_cs[3] = cs[3] + (cs[2] - cs[3]) / np.linalg.norm((cs[2] - cs[3])) * change[3][0]
    new_cs[3] += (cs[3] - cs[0]) / np.linalg.norm((cs[3] - cs[0])) * change[3][1]

    return new_cs


class Drone:
    def __init__(self, img_name):
        self.img_name = img_name

        # 710m * 400m = 16:9 # dia 815m
        # 71 * 40           # dia 81m @ 50m height
        self.max_view = np.array([400, 400])
        self.min_view = np.array([40, 40])

        # open a opencv window and display the initial view
        cv2.namedWindow('navigation viewer')

        # index =  np.where(df['Input.task_image_name'][iii] == pre_rejected_list_df['Input.task_image_name'])[0][0]
        # if pre_rejected_list_df.loc[index,'Reject'] == pre_rejected_list_df.loc[index,'Reject']:
        #     continue
        root_folder_path = '/drone_'
        tif_name = img_name
        self.p_dic = pickle.load(
            open(root_folder_path + 'images/' + tif_name + '.pickle', "rb"))
        print(self.p_dic)
        self.lng_ratio = self.p_dic['lng_ratio']
        self.lat_ratio = self.p_dic['lat_ratio']
        self.gps_botm_left = self.p_dic['gps_botm_left']
        self.gps_top_right = self.p_dic['gps_top_right']
        self.XCoordinate = 0
        self.YCoordinate = 0
        self.newXCoordinate = 0
        self.newYCoordinate = 0
        self.left = False
        self.right = False
        self.up = False
        self.down = False
        self.userXCoordinate = 0
        self.userYCoordinate = 0

        #placeholder until the vln model can supply the coordinates

        

        
        #self.userXCoordinate = input("Enter the x coordinate you want the drone to go to: ")
        #self.userYCoordinate = input("Enter the y coordinate you want the drone to go to: ")

        print(tif_name)
        path = root_folder_path + 'images/' + tif_name
        print(path)
        image = cv2.imread(path)
        cv2.imshow('image', image)
        im_full_map = cv2.imread(path, 1)



        self.im_resized = cv2.resize(im_full_map, (int(im_full_map.shape[1] * self.lng_ratio / self.lat_ratio),
                                                   im_full_map.shape[0]),
                                     interpolation=cv2.INTER_AREA)  # ratio_all = lat_ratio

        im_resized_copy = self.im_resized.copy()

        # load some other basic measurements of the full map

        self.angle = 0
        self.starting_gps = 0.5 * (np.array(self.gps_botm_left) + np.array(self.gps_top_right))

        self.starting_coord = self.gps_to_img_coords(self.starting_gps)

        self.size_of_view = self.min_view
        _gps_coord_top_left = [get_a_gps_coord_at_distance(self.starting_gps[0], self.size_of_view[1] / 2),
                               get_a_gps_coord_at_distance(self.starting_gps[1], -self.size_of_view[0] / 2)]
        _gps_coord_bot_right = [get_a_gps_coord_at_distance(self.starting_gps[0], -self.size_of_view[1] / 2),
                                get_a_gps_coord_at_distance(self.starting_gps[1], self.size_of_view[0] / 2)]

        _im_coords_top_left = self.gps_to_img_coords(_gps_coord_top_left)
        _im_coords_bot_right = self.gps_to_img_coords(_gps_coord_bot_right)

        self.corners = [
            np.array(_im_coords_top_left),
            np.array([_im_coords_bot_right[0], _im_coords_top_left[1]]),
            np.array(_im_coords_bot_right),
            np.array([_im_coords_top_left[0], _im_coords_bot_right[1]])
        ]  # clock wise

        # self.angle = 0
        self.width = 720
        self.height = 720

        self._zoom_speed = 5
        self.step_change_of_view_zoom = np.array([get_a_gps_coord_at_distance(0, self._zoom_speed / 2) / self.lat_ratio,
                                                  get_a_gps_coord_at_distance(0,
                                                                              self._zoom_speed / self.width *
                                                                              self.height / 2) / self.lat_ratio])

        # step_change_of_view_move = self.get_a_gps_coord_at_distance(0, 10)/self.lat_ratio

        # step_change_of_view = np.array(
        #     [self.get_a_gps_coord_at_distance(0,  0.32) / self.lat_ratio, self.get_a_gps_coord_at_distance(0, 0.18) /
        #     self.lat_ratio])
        #
        # step_change_of_view_move = self.get_a_gps_coord_at_distance(0, 0.1) / self.lat_ratio
        self.step_change_angle = 10
        self.dst_pts = np.array([[0, 0],
                                 [self.width - 1, 0],
                                 [self.width - 1, self.height - 1],
                                 [0, self.height - 1]], dtype="float32")

        self.mean_im_coords = np.mean(self.corners, axis=0)
        _corners = [
            self.corners[0] - self.mean_im_coords,
            self.corners[1] - self.mean_im_coords,
            self.corners[2] - self.mean_im_coords,
            self.corners[3] - self.mean_im_coords
        ]  # counter clock wise

        self.rotated_corners = []
        for i in range(4):
            rotated_point = self.mean_im_coords + rotation_anticlock(self.angle, _corners[i])
            if 0 < rotated_point[0] < self.im_resized.shape[1] and 0 < rotated_point[1] < self.im_resized.shape[0]:
                self.rotated_corners.append(rotated_point)
            else:
                break

        self.corners = np.array(self.rotated_corners, dtype="float32")

        # the perspective transformation matrix
        self.M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        self.im_view = cv2.warpPerspective(self.im_resized, self.M, (self.width, self.height))

        self.action_list = []
        self.angle_list = []
        self.pos_list = [self.corners]
        self.attention_list = []
        self.compass_pos = 100
        self.compass_size = 50
        self.count_frame = 0

        cv2.setMouseCallback('navigation viewer', self.click_and_draw)

    def gps_to_img_coords(self, gps):
        return int(round((gps[1] - self.gps_botm_left[1]) / self.lat_ratio)), int(
            round((self.gps_top_right[0] - gps[0]) / self.lat_ratio))

    def img_to_gps_coords(self, img_c):
        return np.array(
            [self.gps_top_right[0] - self.lat_ratio * img_c[1], self.gps_botm_left[1] + self.lat_ratio * img_c[0]])

    def click_and_draw(self, event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(self.angle, _self.angle (y/self.height)*(self.corners[3][1]-self.corners[0][1])*np.sin(-self.angle/ 180 * 3.14159),(x/self.width)*(self.corners[1][0]-self.corners[0][0])*np.sin(self.angle/ 180 * 3.14159), (x/self.width)*(self.corners[1][0]-self.corners[0][0])*np.cos(self.angle/ 180 * 3.14159), (y/self.height)*(self.corners[3][1]-self.corners[0][1])*np.cos(self.angle/ 180 * 3.14159))
            # ((self.corners[1][0]-self.corners[0][0]), (self.corners[3][1]-self.corners[0][1])), self.corners[0],(int((x/self.width)*(self.corners[1][0]-self.corners[0][0])*np.cos(self.angle)+self.corners[0][0]), int((y/self.height)*(self.corners[3][1]-self.corners[0][1])*np.cos(self.angle)+self.corners[0][1]))

            cv2.circle(self.im_resized,
                       (int((x / self.width) * np.linalg.norm(self.corners[1] - self.corners[0]) * np.cos(
                           self.angle / 180 * 3.14159) - (
                                    y / self.height) * np.linalg.norm(self.corners[3] - self.corners[0]) * np.sin(
                           self.angle / 180 * 3.14159) +
                            self.corners[0][0]),
                        int((y / self.height) * np.linalg.norm(self.corners[3] - self.corners[0]) * np.cos(
                            self.angle / 180 * 3.14159) + (
                                    x / self.width) * np.linalg.norm(self.corners[1] - self.corners[0]) * np.sin(
                            self.angle / 180 * 3.14159) +
                            self.corners[0][1])),
                       int(np.linalg.norm(self.corners[1] - self.corners[0]) * 0.09), (0, 255, 0),
                       2)

            self.im_view = cv2.warpPerspective(self.im_resized, self.M, (self.width, self.height))

            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin(-self.angle / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos(-self.angle / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'N',
                        (int(self.compass_pos + self.compass_size * np.sin(-self.angle / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos(-self.angle / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin((-self.angle + 90) / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos((-self.angle + 90) / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'E',
                        (int(self.compass_pos + self.compass_size * np.sin((-self.angle + 90) / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos((-self.angle + 90) / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin((-self.angle + 180) / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos((-self.angle + 180) / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'S',
                        (int(self.compass_pos + self.compass_size * np.sin((-self.angle + 180) / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos((-self.angle + 180) / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin((-self.angle + 270) / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos((-self.angle + 270) / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'W',
                        (int(self.compass_pos + self.compass_size * np.sin((-self.angle + 270) / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos((-self.angle + 270) / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (int(self.width / 2 - 85), int(self.height / 2)),
                     (int(self.width / 2 + 85), int(self.height / 2)),
                     (0, 0, 255), 1)

            cv2.line(self.im_view, (int(self.width / 2), int(self.height / 2 - 85)),
                     (int(self.width / 2), int(self.height / 2 + 85)),
                     (0, 0, 255), 1)

            cv2.imshow('navigation viewer', self.im_view)

            self.attention_list.append(
                [(int((x / self.width) * np.linalg.norm(self.corners[1] - self.corners[0]) * np.cos(
                    self.angle / 180 * 3.14159) - (y / self.height) * np.linalg.norm(
                    self.corners[3] - self.corners[0]) * np.sin(
                    self.angle / 180 * 3.14159) + self.corners[0][0]),
                  int((y / self.height) * np.linalg.norm(self.corners[3] - self.corners[0]) * np.cos(
                      self.angle / 180 * 3.14159) + (x / self.width) * np.linalg.norm(
                      self.corners[1] - self.corners[0]) * np.sin(self.angle / 180 * 3.14159) + self.corners[0][1])),
                 int(np.linalg.norm(self.corners[1] - self.corners[0]) * 0.09)])

    # cv2.setMouseCallback('navigation viewer', click_and_draw)

    def getXCoordinate(self):
       return self.userXCoordinate

    def getYCoordinate(self):
        return self.userYCoordinate

    def move(self, xCoordinate, yCoordinate):
        while True:
            view_ratio = np.linalg.norm(
                self.img_to_gps_coords(self.corners[0]) - self.img_to_gps_coords(self.corners[1])) / (
                                 self.max_view[0] / 11.13 / 1e4)

            step_change_of_view = np.array(
                [get_a_gps_coord_at_distance(0, self._zoom_speed * 10 * view_ratio) / self.lat_ratio,
                 get_a_gps_coord_at_distance(0,
                                             self._zoom_speed /
                                             self.width * self.height * 10 * view_ratio) / self.lat_ratio])

            self.count_frame += 1
            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin(-self.angle / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos(-self.angle / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'N',
                        (int(self.compass_pos + self.compass_size * np.sin(-self.angle / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos(
                             -self.angle / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin((-self.angle + 90) / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos((-self.angle + 90) / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'E',
                        (int(self.compass_pos + self.compass_size * np.sin((-self.angle + 90) / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos((-self.angle + 90) / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin((-self.angle + 180) / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos((-self.angle + 180) / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'S',
                        (int(self.compass_pos + self.compass_size * np.sin((-self.angle + 180) / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos((-self.angle + 180) / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (self.compass_pos, self.compass_pos), (
                int(self.compass_pos + 20 * np.sin((-self.angle + 270) / 180 * 3.14159)),
                int(self.compass_pos - 20 * np.cos((-self.angle + 270) / 180 * 3.14159))),
                     (255, 255, 255), 2)
            cv2.putText(self.im_view, 'W',
                        (int(self.compass_pos + self.compass_size * np.sin((-self.angle + 270) / 180 * 3.14159)),
                         int(self.compass_pos - self.compass_size * np.cos((-self.angle + 270) / 180 * 3.14159))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(self.im_view, (int(self.width / 4), int(self.height / 4)),
                     (int(self.width / 4 + 85), int(self.height / 4)),
                     (0, 0, 255), 1)
            cv2.line(self.im_view, (int(self.width / 4), int(self.height / 4)),
                     (int(self.width / 4), int(self.height / 4 + 85)),
                     (0, 0, 255), 1)

            cv2.line(self.im_view, (int(3 * self.width / 4), int(3 * self.height / 4)),
                     (int(3 * self.width / 4 - 85), int(3 * self.height / 4)),
                     (0, 0, 255), 1)
            cv2.line(self.im_view, (int(3 * self.width / 4), int(3 * self.height / 4)),
                     (int(3 * self.width / 4), int(3 * self.height / 4 - 85)),
                     (0, 0, 255), 1)

            cv2.line(self.im_view, (int(3 * self.width / 4), int(self.height / 4)),
                     (int(3 * self.width / 4 - 85), int(self.height / 4)),
                     (0, 0, 255), 1)
            cv2.line(self.im_view, (int(3 * self.width / 4), int(self.height / 4)),
                     (int(3 * self.width / 4), int(self.height / 4 + 85)),
                     (0, 0, 255), 1)

            cv2.line(self.im_view, (int(self.width / 4), int(3 * self.height / 4)),
                     (int(self.width / 4 + 85), int(3 * self.height / 4)),
                     (0, 0, 255), 1)
            cv2.line(self.im_view, (int(self.width / 4), int(3 * self.height / 4)),
                     (int(self.width / 4), int(3 * self.height / 4 - 85)),
                     (0, 0, 255), 1)

            cv2.line(self.im_view, (int(self.width / 2 - 85), int(self.height / 2)),
                     (int(self.width / 2 + 85), int(self.height / 2)),
                     (0, 0, 255), 1)

            cv2.line(self.im_view, (int(self.width / 2), int(self.height / 2 - 85)),
                     (int(self.width / 2), int(self.height / 2 + 85)),
                     (0, 0, 255), 1)

            cv2.imshow('navigation viewer', self.im_view)

            #currently this is needed to show the navigation viewer - why?
            event = cv2.waitKey(0)

            user = input("command: ")
            farthest_nearest = ""

            if "nearest" in user:
                farthest_nearest = "nearest"
            elif "farthest" in user:
                farthest_nearest = "farthest"

            distance = user.split()[0]
            distance = str(distance)
            object_name = user.split()[1]
            object_name = str(object_name)
            #print(type(object_name))

            #print(type(self.XCoordinate))

            parser = argparse.ArgumentParser()
            # Get data configuration

            parser.add_argument('-image_folder', type=str, default='/drone_images/579.tif')
            parser.add_argument('-output_folder', type=str, default='./output_xview', help='path to outputs')
            cuda = False  # torch.cuda.is_available()

            parser.add_argument('-plot_flag', type=bool, default=True)
            parser.add_argument('-secondary_classifier', type=bool, default=False)
            parser.add_argument('-cfg', type=str, default='/xview-yolov3/cfg/c60_a30symmetric.cfg', help='cfg file path')
            parser.add_argument('-class_path', type=str, default='/xview-yolov3/data/xview.names', help='path to class label file')
            parser.add_argument('-conf_thres', type=float, default=0.99, help='object confidence threshold')
            parser.add_argument('-nms_thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
            parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
            parser.add_argument('-img_size', type=int, default=32 * 51, help='size of each image dimension')
            opt = parser.parse_args()



            self.XCoordinate = float((self.corners[0][0] + self.corners[1][0]) / 2)
            self.YCoordinate = float((self.corners[1][1] + self.corners[2][1]) / 2)

            print(type(self.XCoordinate))
            print(self.XCoordinate)
            self.userXCoordinate = detect(opt,1,object_name,distance,self.XCoordinate,self.YCoordinate)
            self.userYCoordinate = detect(opt,2,object_name,distance,self.XCoordinate,self.YCoordinate)

            print(self.userYCoordinate)

            print("CurrX: " + str(self.XCoordinate) + "   CurrY: " + str(self.YCoordinate))

            # until vln is ready, the following user actions allows testing of drone movements using
            # keys or the user-provided coordinates
            decision = input("would you like to follow coordinates (y/n): ")
            if decision == 'y':
                self.XCoordinate = (self.corners[0][0] + self.corners[1][0]) / 2
                self.YCoordinate = (self.corners[1][1] + self.corners[2][1]) / 2

                # print(self.XCoordinate)
                # print(self.YCoordinate)
                # print(newYCoordinate)

                if int(xCoordinate) < int(self.XCoordinate):
                    self.left = True
                    self.right = False
                else:
                    self.left = False
                    self.right = True
                if int(yCoordinate) < int(self.YCoordinate):
                    self.up = True
                    self.down = False
                else:
                    self.up = False
                    self.down = True

                #move the drone in steps until it has reached the intended coordinates
                while (self.left == True or self.right == True or self.up == True or self.down == True):
                    self.XCoordinate = (self.corners[0][0] + self.corners[1][0]) / 2
                    self.YCoordinate = (self.corners[1][1] + self.corners[2][1]) / 2

                    if self.left == True and int(self.userXCoordinate) > int(self.XCoordinate):
                        self.left = False
                    if self.right == True and int(self.userXCoordinate) < int(self.XCoordinate):
                        self.right = False
                    if self.up == True and int(self.userYCoordinate) > int(self.YCoordinate):
                        self.up = False
                    if self.down == True and int(self.userYCoordinate) < int(self.YCoordinate) :
                        self.down = False

                    print("CurrX: " + str(self.XCoordinate) + "   CurrY: " + str(self.YCoordinate) + "    NewX: " + str(xCoordinate) + "    NewY: " + str(yCoordinate))
                    print("Left: " + str(self.left) + "    Right: " + str(self.right) + "    Up: "  + str(self.up) + "      Down: " + str(self.down))


                    if self.left == True:
                        print('*****************************************************')
                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([-1, 0]), step_change_of_view * np.array([-1, 0]),
                             step_change_of_view * np.array([-1, 0]), step_change_of_view * np.array([-1, 0])]
                        )
                        new_corners = []
                        for i in _new_corners:
                            # print("i[0] " + str(i[0]) + "  i[1] " + str(i[1]))
                            # print("self.im_resized.shape[1] " + str(self.im_resized.shape[1]) + "  self.im_resized.shape[0] " + str(self.im_resized.shape[0]))
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        # print("left==True " + str(len(new_corners)) + "  __new_corners " + str(len(_new_corners)))
                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif self.right == True:
                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([1, 0]), step_change_of_view * np.array([1, 0]),
                             step_change_of_view * np.array([1, 0]), step_change_of_view * np.array([1, 0])]
                        )
                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif self.up == True:
                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([0, -1]), step_change_of_view * np.array([0, -1]),
                             step_change_of_view * np.array([0, -1]), step_change_of_view * np.array([0, -1])]
                        )

                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif self.down == True:
                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([0, 1]), step_change_of_view * np.array([0, 1]),
                             step_change_of_view * np.array([0, 1]), step_change_of_view * np.array([0, 1])]
                        )
                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    self.angle_list.append(self.angle)
                    self.pos_list.append(self.corners)

            else: # move based on keystrokes
                for i in range(0, 71): # so we can zoom all the way out with one key
                    if event == 27:
                        approve = ''
                        print('\n====== You have pressed ESC ======')
                        break

                    elif event == ord('2'):

                        _new_corners = change_corner(
                            self.corners,
                            [self.step_change_of_view_zoom * np.array([-1, -1]),
                             self.step_change_of_view_zoom * np.array([1, -1]),
                             self.step_change_of_view_zoom * np.array([1, 1]),
                             self.step_change_of_view_zoom * np.array([-1, 1])]
                        )
                        if np.linalg.norm(self.img_to_gps_coords(_new_corners[0]) - self.img_to_gps_coords(_new_corners[1])) > \
                                self.max_view[0] / \
                                11.13 / 1e4:
                            continue

                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif event == ord('1'):
                        _new_corners = change_corner(
                            self.corners,
                            [self.step_change_of_view_zoom * np.array([1, 1]),
                             self.step_change_of_view_zoom * np.array([-1, 1]),
                             self.step_change_of_view_zoom * np.array([-1, -1]),
                             self.step_change_of_view_zoom * np.array([1, -1])]
                        )

                        if np.linalg.norm(self.img_to_gps_coords(_new_corners[0]) - self.img_to_gps_coords(_new_corners[1])) < \
                                self.min_view[0] / \
                                11.13 / 1e4:
                            continue

                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif event == ord('w'):

                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([0, -1]), step_change_of_view * np.array([0, -1]),
                             step_change_of_view * np.array([0, -1]), step_change_of_view * np.array([0, -1])]
                        )

                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))


                    elif event == ord('s'):

                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([0, 1]), step_change_of_view * np.array([0, 1]),
                             step_change_of_view * np.array([0, 1]), step_change_of_view * np.array([0, 1])]
                        )
                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))
                    elif event == ord('d'):

                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([1, 0]), step_change_of_view * np.array([1, 0]),
                             step_change_of_view * np.array([1, 0]), step_change_of_view * np.array([1, 0])]
                        )
                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif event == ord('a'):

                        _new_corners = change_corner(
                            self.corners,
                            [step_change_of_view * np.array([-1, 0]), step_change_of_view * np.array([-1, 0]),
                             step_change_of_view * np.array([-1, 0]), step_change_of_view * np.array([-1, 0])]
                        )
                        new_corners = []
                        for i in _new_corners:
                            if 0 < i[0] < self.im_resized.shape[1] and 0 < i[1] < self.im_resized.shape[0]:
                                new_corners.append(i)
                            else:
                                break

                        if len(new_corners) != 4:
                            continue
                        else:
                            self.corners = np.array(new_corners, dtype="float32")

                            # the perspective transformation matrix
                            M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                            # directly warp the rotated rectangle to get the straightened rectangle
                            self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif event == ord('e'):
                        self.angle += self.step_change_angle

                        self.mean_im_coords = np.mean(self.corners, axis=0)
                        _corners = [
                            self.corners[0] - self.mean_im_coords,
                            self.corners[1] - self.mean_im_coords,
                            self.corners[2] - self.mean_im_coords,
                            self.corners[3] - self.mean_im_coords
                        ]  # counter clock wise

                        self.rotated_corners = []
                        for i in range(4):
                            rotated_point = self.mean_im_coords + rotation_anticlock(self.step_change_angle, _corners[i])
                            if 0 < rotated_point[0] < self.im_resized.shape[1] and 0 < rotated_point[1] < self.im_resized.shape[
                                0]:
                                self.rotated_corners.append(rotated_point)
                            else:
                                break

                        if len(self.rotated_corners) != 4:
                            self.angle -= self.step_change_angle
                            continue
                        self.corners = np.array(self.rotated_corners, dtype="float32")

                        # the perspective transformation matrix
                        M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                        # directly warp the rotated rectangle to get the straightened rectangle
                        self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    elif event == ord('q'):
                        self.angle -= self.step_change_angle

                        self.mean_im_coords = np.mean(self.corners, axis=0)
                        _corners = [
                            self.corners[0] - self.mean_im_coords,
                            self.corners[1] - self.mean_im_coords,
                            self.corners[2] - self.mean_im_coords,
                            self.corners[3] - self.mean_im_coords
                        ]  # counter clock wise

                        self.rotated_corners = []
                        for i in range(4):
                            rotated_point = self.mean_im_coords + rotation_anticlock(-self.step_change_angle, _corners[i])
                            if 0 < rotated_point[0] < self.im_resized.shape[1] and 0 < rotated_point[1] < self.im_resized.shape[
                                0]:
                                self.rotated_corners.append(rotated_point)
                            else:
                                break

                        if len(self.rotated_corners) != 4:
                            self.angle += self.step_change_angle
                            continue
                        self.corners = np.array(self.rotated_corners, dtype="float32")

                        # the perspective transformation matrix
                        M = cv2.getPerspectiveTransform(self.corners, self.dst_pts)

                        # directly warp the rotated rectangle to get the straightened rectangle
                        self.im_view = cv2.warpPerspective(self.im_resized, M, (self.width, self.height))

                    self.angle_list.append(self.angle)
                    self.pos_list.append(self.corners)

            #print(end)
        # set the push button feed back record the actions

        # set an end button and compare/save the route