import argparse
import os
import time
import cv2
import numpy as np
import torch
from src.model import IWPODNet
from src.utils import *
from src.label import *
from src.projection_utils import *


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)


def reconstruct_new(Iorig, I, Y, out_size, threshold=.9):
    net_stride = 2 ** 4
    side = ((208. + 40.) / 2.) / net_stride  # based on rescaling of training data

    Probs = Y[0, ...].cpu().numpy()
    Affines = Y[-6:, ...].cpu().numpy()  # gets the last six coordinates related to the Affine transform
    rx, ry = Y.shape[1:]

    #
    #  Finds cells with classification probability greater than threshold
    #
    xx, yy = np.where(Probs > threshold)
    WH = getWH(I.shape)
    MN = WH / net_stride

    #
    #  Warps canonical square to detected LP
    #
    vxx = vyy = 0.5  # alpha -- must match training script
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
    labels = []

    for i in range(len(xx)):
        y, x = xx[i], yy[i]
        affine = Affines[:, y, x]
        prob = Probs[y, x]

        mn = np.array([float(x) + .5, float(y) + .5])

        #
        #  Builds affine transformatin matrix
        #
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0.)
        A[1, 1] = max(A[1, 1], 0.)

        pts = np.array(A * base(vxx, vyy))  # *alpha
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2, 1))

        pts_prop = pts_MN / MN.reshape((2, 1))

        labels.append(DLabel(0, pts_prop, prob))

    final_labels = nms(labels, .1)
    TLps = []  # list of detected plates

    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for i, label in enumerate(final_labels):
            ptsh = np.concatenate((label.pts * getWH(Iorig.shape).reshape((2, 1)), np.ones((1, 4))))
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            H = find_T_matrix(ptsh, t_ptsh)
            Ilp = cv2.warpPerspective(Iorig, H, out_size, flags=cv2.INTER_CUBIC, borderValue=.0)
            TLps.append(Ilp)
    return final_labels, TLps


def detect_lp_width(model, I, MAXWIDTH, net_step, out_size, threshold):
    #
    #  Resizes input image and run IWPOD-NET
    #

    # Computes resize factor
    factor = min(1, MAXWIDTH / I.shape[1])
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()

    # dimensions must be multiple of the network stride
    w += (w % net_step != 0) * (net_step - w % net_step)
    h += (h % net_step != 0) * (net_step - h % net_step)

    # resizes image
    Iresized = cv2.resize(I, (w, h), interpolation=cv2.INTER_CUBIC)
    T = Iresized.copy()

    # Prepare to feed to IWPOD-NET
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        model.eval()
        inputs = torch.from_numpy(T).permute(2, 0, 1).float()
        inputs = torch.unsqueeze(inputs, dim=0).to(device)
        start = time.time()
        outputs = model(inputs)
        Yr = torch.squeeze(outputs)
        elapsed = time.time() - start

        L, TLps = reconstruct_new(I, Iresized, Yr, out_size, threshold)

    return L, TLps, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default=os.path.join('images', 'example_aolp_fullimage.jpg'), help='Input Image')
    parser.add_argument('-v', '--vtype', type=str, default='fullimage', help='Image type (car, truck, bus, bike or fullimage)')
    parser.add_argument('-t', '--lp_threshold', type=float, default=0.35, help='Detection Threshold')
    args = parser.parse_args()

    lp_threshold = args.lp_threshold
    ocr_input_size = [80, 240]  # desired LP size (width x height)

    Ivehicle = cv2.imread(args.image)
    vtype = args.vtype
    iwh = np.array(Ivehicle.shape[1::-1], dtype=float).reshape((2, 1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mymodel = IWPODNet()
    mymodel.load_state_dict(torch.load('weights/iwpodnet_retrained_epoch10000.pth', map_location=device)['model_state_dict'])
    mymodel.to(device)

    if vtype in ['car', 'bus', 'truck']:
        #
        #  Defines crops for car, bus, truck based on input aspect ratio (see paper)
        #
        ASPECTRATIO = max(1.0, min(2.75, 1.0 * Ivehicle.shape[1] / Ivehicle.shape[0]))  # width over height
        WPODResolution = 256  # faster execution
        lp_output_resolution = tuple(ocr_input_size[::-1])
    elif vtype == 'fullimage':
        #
        #  Defines crop if vehicles were not cropped
        #
        ASPECTRATIO = 1.0
        WPODResolution = 480  # larger if full image is used directly
        lp_output_resolution = tuple(ocr_input_size[::-1])
    else:
        #
        #  Defines crop for motorbike
        #
        ASPECTRATIO = 1.0  # width over height
        WPODResolution = 208
        lp_output_resolution = (int(1.5 * ocr_input_size[0]), ocr_input_size[0])  # for bikes, the LP aspect ratio is lower

    Llp, LlpImgs, _ = detect_lp_width(mymodel, im2single(Ivehicle), WPODResolution * ASPECTRATIO, 2 ** 4, lp_output_resolution, lp_threshold)

    for i, img in enumerate(LlpImgs):
        #
        #  Draws LP quadrilateral in input image
        #
        pts = Llp[i].pts * iwh
        draw_losangle(Ivehicle, pts, color=(0, 0, 255.), thickness=2)
        #
        #  Shows each detected LP
        #
        cv2.imshow('Rectified plate %d' % i, img)
        cv2.waitKey()

    #
    #  Shows original image with deteced plates (quadrilateral)
    #
    cv2.imshow('Image and LPs', Ivehicle)
    cv2.waitKey()
    cv2.destroyAllWindows()