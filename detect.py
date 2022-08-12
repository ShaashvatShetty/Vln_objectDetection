import argparse
import time
#import tensorflow as tf
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
import nltk
import numpy
from nltk.corpus import wordnet
import math
import sys
targets_path = '/xview-yolov3/utils/targets_c60.mat'

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
print(opt)

def detect(opt, state = 0, aa = '', bb= '', cc = 0, dd = 0):



    if opt.plot_flag:
        #os.system('rm -rf ' + opt.output_folder + '_img')
        os.makedirs(opt.output_folder + '_img', exist_ok=True)
    #os.system('rm -rf ' + opt.output_folder)
    os.makedirs(opt.output_folder, exist_ok=True)
    device = torch.device('cuda:0' if cuda else 'cpu')

    # Load model 1
    model = Darknet(opt.cfg, opt.img_size)
    checkpoint = torch.load('/Downloads/best_weights_obj_detec_aerial.pt', map_location='cpu')

    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    del checkpoint

    # current = model.state_dict()
    # saved = checkpoint['model']
    # # 1. filter out unnecessary keys
    # saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
    # # 2. overwrite entries in the existing state dict
    # current.update(saved)
    # # 3. load the new state dict
    # model.load_state_dict(current)
    # model.to(device).eval()
    # del checkpoint, current, saved

    # Load model 2
    if opt.secondary_classifier:
        model2 = ConvNetb()
        checkpoint = torch.load('weights/classifier.pt', map_location='cpu')

        model2.load_state_dict(checkpoint['model'])
        model2.to(device).eval()
        del checkpoint

        # current = model2.state_dict()
        # saved = checkpoint['model']
        # # 1. filter out unnecessary keys
        # saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
        # # 2. overwrite entries in the existing state dict
        # current.update(saved)
        # # 3. load the new state dict
        # model2.load_state_dict(current)
        # model2.to(device).eval()
        # del checkpoint, current, saved
    else:
        model2 = None

    # Set Dataloader
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    dataloader = ImageFolder(opt.image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

    #print(f"this is dataloader {dataloader}")

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    detections = None
    mat_priors = scipy.io.loadmat(targets_path)
    for batch_i, (img_paths, img) in enumerate(dataloader):
        print('\n', batch_i, img.shape, end=' ')

        img_ud = np.ascontiguousarray(np.flip(img, axis=1))
        img_lr = np.ascontiguousarray(np.flip(img, axis=2))

        preds = []
        length = opt.img_size
        ni = int(math.ceil(img.shape[1] / length))  # up-down
        nj = int(math.ceil(img.shape[2] / length))  # left-right
        for i in range(ni):  # for i in range(ni - 1):
            print('row %g/%g: ' % (i, ni), end='')

            for j in range(nj):  # for j in range(nj if i==0 else nj - 1):
                print('%g ' % j, end='', flush=True)

                # forward scan
                y2 = min((i + 1) * length, img.shape[1])
                y1 = y2 - length
                x2 = min((j + 1) * length, img.shape[2])
                x1 = x2 - length

                # Get detections
                with torch.no_grad():
                    # Normal orientation
                    chip = torch.from_numpy(img[:, y1:y2, x1:x2]).unsqueeze(0).to(device)
                    pred = model(chip)
                    pred = pred[pred[:, :, 4] > opt.conf_thres]
                    # if (j > 0) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 0] - pred[:, 2] / 2 > 2)]  # near left border
                    # if (j < nj) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 0] + pred[:, 2] / 2 < 606)]  # near right border
                    # if (i > 0) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 1] - pred[:, 3] / 2 > 2)]  # near top border
                    # if (i < ni) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 1] + pred[:, 3] / 2 < 606)]  # near bottom border
                    if len(pred) > 0:
                        pred[:, 0] += x1
                        pred[:, 1] += y1
                        preds.append(pred.unsqueeze(0))

                    # # Flipped Up-Down
                    # chip = torch.from_numpy(img_ud[:, y1:y2, x1:x2]).unsqueeze(0).to(device)
                    # pred = model(chip)
                    # pred = pred[pred[:, :, 4] > opt.conf_thres]
                    # if len(pred) > 0:
                    #     pred[:, 0] += x1
                    #     pred[:, 1] = img.shape[1] - (pred[:, 1] + y1)
                    #     preds.append(pred.unsqueeze(0))

                    # # Flipped Left-Right
                    # chip = torch.from_numpy(img_lr[:, y1:y2, x1:x2]).unsqueeze(0).to(device)
                    # pred = model(chip)
                    # pred = pred[pred[:, :, 4] > opt.conf_thres]
                    # if len(pred) > 0:
                    #     pred[:, 0] = img.shape[2] - (pred[:, 0] + x1)
                    #     pred[:, 1] += y1
                    #     preds.append(pred.unsqueeze(0))

        if len(preds) > 0:
            detections = non_max_suppression(torch.cat(preds, 1), opt.conf_thres, opt.nms_thres, mat_priors, img,
                                             model2, device)
            img_detections.extend(detections)
            imgs.extend(img_paths)

        print('Batch %d... (Done %.3fs)' % (batch_i, time.time() - prev_time))
        prev_time = time.time()

    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    if len(img_detections) == 0:
        return
    #stores the objects and its number
    detected_objs = {}
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("image %g: '%s'" % (img_i, path))

        if opt.plot_flag:
            img = cv2.imread(path)

        # # The amount of padding that was added
        # pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        # pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # # Image height and width after padding is removed
        # unpad_h = opt.img_size - pad_y
        # unpad_w = opt.img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(color_list, len(unique_classes))

            # write results to .txt file
            results_path = os.path.join(opt.output_folder, path.split('/')[-1])
            if os.path.isfile(results_path + '.txt'):
                os.remove(results_path + '.txt')

            results_img_path = os.path.join(opt.output_folder + '_img', path.split('/')[-1])
            with open(results_path.replace('.bmp', '.tif') + '.txt', 'a') as file:
                for i in unique_classes:
                    n = (detections[:, -1].cpu() == i).sum()
                   # print('%g %ss' % (n, classes[int(i)]))

                    

                    detected_objs[classes[int(i)].lower()] = int(n) 


                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # Rescale coordinates to original dimensions
                    # box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    # box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    # y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0]).round().item()
                    # x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round().item()
                    # x2 = (x1 + box_w).round().item()
                    # y2 = (y1 + box_h).round().item()
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)


                    # write to file
                    xvc = xview_indices2classes(int(cls_pred))  # xview class
                    # if (xvc != 21) & (xvc != 72):
                    file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, xvc, cls_conf * conf))

                    ###############
                    def plot_one_box(x, img, color=None, label=None, line_thickness=None,center=None):
                        # Plots one bounding box on image img
                        tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                        #color = color or [random.randint(0, 255) for _ in range(3)]
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        #cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
                        if label:
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(label, 0, fontScale=tl / 5, thickness=tf)[0]
                            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                            
                            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                        if center:
                            font_thickness = max(tl - 1, 1)
                            text_size = cv2.getTextSize(label, 0, fontScale=tl / 5, thickness=tf-3)[0]
                            cv2.putText(img, center, (c1[0], c1[1] +8), 0, tl / 5, [255,255,255], thickness=font_thickness, lineType=cv2.LINE_AA) 

                    if opt.plot_flag:
                        # Add the bbox to the plot
                        label = '%s %.2f' % (classes[int(cls_pred)], cls_conf) if cls_conf > 0.05 else None
                        color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                        centerX = (x1+x2)/2
                        centerX = centerX.numpy()
                        centerX = numpy.round(centerX)
                        centerX = int(centerX)
                        string_centerX = str(centerX)

                        centerY = (y1+y2)/2
                        centerY = centerY.numpy()
                        centerY = numpy.round(centerY)
                        centerY = int(centerY)

                        string_centerY = str(centerY)

                        centerPoint = "("+string_centerX+","+string_centerY+")" 
                        plot_one_box([x1, y1, x2, y2], img, label=label, color=color, line_thickness=1.5,center = centerPoint)

            if opt.plot_flag:
                # Save generated image with detections
                cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)
                print(results_img_path)
                #print(detected_objs)
                #print(centerPoint)
                #print(centerX.numpy())
    #######################################
    import re
    from collections import Counter

    WORD = re.compile(r"\w+")


    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator


    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)


    

    label_store2 = []
    def get_object_coord(input_obj,farthest_nearest,x,y):
        label_store = []
        item_store = []
        needed_store ={}
        nearest = sys.maxsize
        nearestX = 0
        nearestY = 0

        farthestX = 0
        farthestY = 0
        label2 = ""
        farthest = -1
        #input_obj= input("Searching for number of: ").lower()
        #print(wordnet.synsets(input_obj))
        sys_input_obj = wordnet.synset(wordnet.synsets(input_obj)[0].name())

        for item in detected_objs:
            temp = []
            #item_store.append(item)
            for i in  [''.join(c for c in s if c.isalpha()) for s in item.split()]:
                try:
                    sys_item =wordnet.synset(wordnet.synsets(i)[0].name())
                    temp.append(sys_input_obj.wup_similarity(sys_item))
                except:
                    continue

            if len(temp) > 0 and sum(temp)/len(temp) >= 0.75 or get_cosine(text_to_vector(input_obj),text_to_vector(item)) >=0.8:
                print(f'{item} = {detected_objs[item]}  word similarity={sum(temp)/len(temp)}  cosine similarity={get_cosine(text_to_vector(input_obj),text_to_vector(item))}')
                item_store.append(item)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
                    x_coord = (x1+x2)/2
                    x_coord = x_coord.numpy()
                    x_coord = str(x_coord)

                    y_coord = (y1+y2)/2
                    y_coord = y_coord.numpy()
                    y_coord = str(y_coord)


                    label = '%s' % (classes[int(cls_pred)])
                    label = label.lower()
                    label_store2.append(label)
                    if label in item_store:
                        #print(f'{label} located at {x_coord}  {y_coord}')
                        if farthest_nearest == "nearest":
                            
                            if math.dist([x,y],[float(x_coord),float(y_coord)])<nearest:
                                nearest = math.dist([x,y],[float(x_coord),float(y_coord)])
                                nearestX = x_coord
                                nearestY = y_coord
                                label2 = label
                        if farthest_nearest == "farthest":
                            if math.dist([x,y],[float(x_coord),float(y_coord)])>farthest:
                                farthest = math.dist([x,y],[float(x_coord),float(y_coord)])
                                farthestX = x_coord
                                farthestY = y_coord
                                label2 = label
        if farthest_nearest =="nearest":                        
            print(f"nearest {label2} at ({nearestX},{nearestY})")

        if farthest_nearest == "farthest":
            print(f"farthest {label2} at ({farthestX},{farthestY}")
                    #label_store.append(label)
                    
    def getX(input_obj,farthest_nearest,x,y):
        label_store = []
        item_store = []
        needed_store ={}
        nearest = sys.maxsize
        nearestX = 0
        nearestY = 0

        farthestX = 0
        farthestY = 0
        label2 = ""
        farthest = -1
        sys_input_obj = wordnet.synset(wordnet.synsets(input_obj)[0].name())

        for item in detected_objs:
            temp = []
            #item_store.append(item)
            for i in  [''.join(c for c in s if c.isalpha()) for s in item.split()]:
                try:
                    sys_item =wordnet.synset(wordnet.synsets(i)[0].name())
                    temp.append(sys_input_obj.wup_similarity(sys_item))
                except:
                    continue

            if len(temp) > 0 and sum(temp)/len(temp) >= 0.85 or get_cosine(text_to_vector(input_obj),text_to_vector(item)) >=0.7:
                #print(f'{item} = {detected_objs[item]}  word similarity={sum(temp)/len(temp)}  cosine similarity={get_cosine(text_to_vector(input_obj),text_to_vector(item))}')
                item_store.append(item)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
                    x_coord = (x1+x2)/2
                    x_coord = x_coord.numpy()
                    x_coord = str(x_coord)
                    y_coord = (y1+y2)/2
                    y_coord = y_coord.numpy()
                    y_coord = str(y_coord)

                    label = '%s' % (classes[int(cls_pred)])
                    label = label.lower()
                    #label_store2.append(label)
                    if label in item_store:
                        #print(f'{label} located at {x_coord}  {y_coord}')
                        if farthest_nearest == "nearest":
                            
                            if math.dist([x,y],[float(x_coord),float(y_coord)])<nearest:
                                nearest = math.dist([x,y],[float(x_coord),float(y_coord)])
                                nearestX = x_coord
                                nearestY = y_coord
                                label2 = label
                        if farthest_nearest == "farthest":
                            if math.dist([x,y],[float(x_coord),float(y_coord)])>farthest:
                                farthest = math.dist([x,y],[float(x_coord),float(y_coord)])
                                farthestX = x_coord
                                farthestY = y_coord
                                #label2 = label
        if farthest_nearest =="nearest":
            #print(nearestX)                        
            return float(nearestX)

        if farthest_nearest == "farthest":
            return float(farthestX)
    def getY(input_obj,farthest_nearest,x,y):
        label_store = []
        item_store = []
        needed_store ={}
        nearest = sys.maxsize
        nearestX = 0
        nearestY = 0

        farthestX = 0
        farthestY = 0
        label2 = ""
        farthest = -1
        sys_input_obj = wordnet.synset(wordnet.synsets(input_obj)[0].name())

        for item in detected_objs:
            temp = []
            #item_store.append(item)
            for i in  [''.join(c for c in s if c.isalpha()) for s in item.split()]:
                try:
                    sys_item =wordnet.synset(wordnet.synsets(i)[0].name())
                    temp.append(sys_input_obj.wup_similarity(sys_item))
                except:
                    continue

            if len(temp) > 0 and sum(temp)/len(temp) >= 0.85 or get_cosine(text_to_vector(input_obj),text_to_vector(item)) >=0.7:
                #print(f'{item} = {detected_objs[item]}  word similarity={sum(temp)/len(temp)}  cosine similarity={get_cosine(text_to_vector(input_obj),text_to_vector(item))}')
                item_store.append(item)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
                    x_coord = (x1+x2)/2
                    x_coord = x_coord.numpy()
                    x_coord = str(x_coord)
                    y_coord = (y1+y2)/2
                    y_coord = y_coord.numpy()
                    y_coord = str(y_coord)

                    label = '%s' % (classes[int(cls_pred)])
                    label = label.lower()
                    #label_store2.append(label)
                    if label in item_store:
                        #print(f'{label} located at {x_coord}  {y_coord}')
                        if farthest_nearest == "nearest":
                            
                            if math.dist([x,y],[float(x_coord),float(y_coord)])<nearest:
                                nearest = math.dist([x,y],[float(x_coord),float(y_coord)])
                                nearestX = x_coord
                                nearestY = y_coord
                                label2 = label
                        if farthest_nearest == "farthest":
                            if math.dist([x,y],[float(x_coord),float(y_coord)])>farthest:
                                farthest = math.dist([x,y],[float(x_coord),float(y_coord)])
                                farthestX = x_coord
                                farthestY = y_coord
                                #label2 = label
        if farthest_nearest =="nearest":
            #print(nearestY)                        
            return float(nearestY)

        if farthest_nearest == "farthest":
            return float(farthestY)
    

    if state == 1:
        res_x = getX(aa, bb, cc, dd)
        return res_x
    elif state == 2:
        res_y=getY(aa, bb, cc, dd)
        return res_y
    else:

        farthest_nearest = ""
        obj = ""
        user = input("command: ")

        if "nearest" in user:
            farthest_nearest = "nearest"
        elif "farthest" in user:
            farthest_nearest = "farthest"

        distance = user.split()[0]
        object_name = user.split()[1]


        #x = input("pick an object: ")
        get_object_coord(object_name,distance,0,0)
        getX(object_name,distance,0,0)
        getY(object_name,distance,0,0)
        return detected_objs
    #get_object_coord(x,farthest_nearest,0,0)

'''
    if opt.plot_flag:
        from scoring import score
        score.score(opt.output_folder + '/', '/Users/glennjocher/Downloads/DATA/xview/xView_train.geojson', '.')
'''

class ConvNetb(nn.Module):
    def __init__(self, num_classes=60):
        super(ConvNetb, self).__init__()
        n = 64  # initial convolution size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 2),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 4),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 8),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(n * 8, n * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 16),
            nn.LeakyReLU())
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(n * 16, n * 32, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(n * 32),
        #     nn.LeakyReLU())

        # self.fc = nn.Linear(int(8192), num_classes)  # 64 pixels, 4 layer, 64 filters
        self.fully_conv = nn.Conv2d(n * 16, 60, kernel_size=4, stride=1, padding=0, bias=True)

    def forward(self, x):  # 500 x 1 x 64 x 64
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.layer6(x)
        # x = self.fc(x.reshape(x.size(0), -1))
        x = self.fully_conv(x)
        return x.squeeze()  # 500 x 60


if __name__ == '__main__':
    torch.cuda.empty_cache()
    res = detect(opt)
    print(res)

    ###############
    # store
    ##############
    torch.cuda.empty_cache()
