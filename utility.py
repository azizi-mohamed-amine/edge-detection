# -*- coding: utf-8 -*-
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import cv2
import math
import numpy as np


def is_near(pt1, pt2, npt):
    x1, y1 = pt1
    x2, y2 = pt2
    dx12 = x2 - x1
    dy12 = y2 - y1
    dl12 = math.sqrt(dx12 * dx12 + dy12 * dy12)
    dx12 /= dl12
    dy12 /= dl12

    x, y = npt
    dx = x - x1
    dy = y - y1

    projlen = dx * dx12 + dy * dy12

    xp = x1 + dx12 * projlen
    yp = y1 + dy12 * projlen

    dppx = xp - x
    dppy = yp - y

    dist = math.sqrt(dppx * dppx + dppy * dppy)
    if dppx < 0:
        dist = -dist
    return dist

def imgradientxy(frame, v_dir):
    imgDiffX = cv2.Sobel(frame, cv2.CV_32F, 1, 0, 3)
    imgDiffY = cv2.Sobel(frame, cv2.CV_32F, 0, 1, 3)

    Gm = imgDiffX * v_dir[0] + imgDiffY * v_dir[1]
    return Gm

def get2Point(x0, y0, ki, xd, yd, vdir, imH):
    x1 = 0
    y1 = y0 - x0 * ki
    if y1 < 0:
        x1 = x0 - y0 / ki
        y1 = 0

    rd = vdir[1] / vdir[0]

    x2 = ((yd-y0) + ki*x0 - rd*xd) / (ki-rd)
    y2 = y0 + ki*(x2-x0)

    if y2 > imH - 1:
        y2 = imH - 1
        x2 = x0 + (y2-y0) / ki

    return [x1, y1, x2, y2]




def setLineScore(canny_img, gm_img, lines):

    scores1 = []
    scores2 = []
    for l in lines:
        pt1 = l[0]
        pt2 = l[1]
        dp = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
        lendp = math.sqrt(dp[0]*dp[0] + dp[1]*dp[1])
        dp = [dp[0]/lendp, dp[1]/lendp]
        nlen = int(lendp / 2)
        score1 = 0
        score2 = 0
        for i in range(nlen):
            pt = [int(0.5+pt1[0]+2.0*i*dp[0]), int(0.5+pt1[1]+2.0*i*dp[1])]
            cval = canny_img[pt[1]][pt[0]]
            score1 += cval / 255
            cval = gm_img[pt[1]][pt[0]]
            score2 += cval

        scores1.append(score1)
        scores2.append(score2)
    return scores1, scores2
def get_dist_score(d1, d01, d2, d02):
    dd1 = d1/d01 - 1
    dd2 = d2/d02 - 1

    res = math.exp(-(dd1*dd1 + dd2*dd2))
    return res;
def get_view_rect(p1, p2, v_dir, r1, r2, height, width):
    ps1 = [p1[0]+v_dir[0]*0.5*r1, p1[1]+v_dir[1]*0.5*r1]
    ps2 = [p2[0]+v_dir[0]*0.5*r2, p2[1]+v_dir[1]*0.5*r2]
    pe1 = [p1[0]+v_dir[0]*2.2*r1, p1[1]+v_dir[1]*2.2*r1]
    pe2 = [p2[0]+v_dir[0]*2.2*r2, p2[1]+v_dir[1]*2.2*r2]

    dls = [ps2[0]-ps1[0], ps2[1]-ps1[1]]
    lends = math.sqrt(dls[0]*dls[0] + dls[1]*dls[1])
    dls = [dls[0] / lends, dls[1] / lends]
    ps2 = [ps2[0]-dls[0]*r2, ps2[1]-dls[1]*r2]
    xs = 0
    ys = ps1[1] - ps1[0] * dls[1] / dls[0]
    if ys < 0:
        ys = 0
        xs = ps1[0] - ps1[1] * dls[0] / dls[1]
    ps1 = [xs, ys]



    dls = [pe2[0]-pe1[0], pe2[1]-pe1[1]]
    lends = math.sqrt(dls[0]*dls[0] + dls[1]*dls[1])
    dls = [dls[0]/lends, dls[1]/lends]
    pe2 = [pe2[0] - dls[0] * r2, pe2[1] - dls[1] * r2]
    xs = 0
    ys = pe1[1] - pe1[0] * dls[1] / dls[0]
    if ys < 0:
        ys = 0
        xs = pe1[0] - pe1[1] * dls[0] / dls[1]
    pe1 = [xs, ys]

    img = np.zeros((height, width), np.uint8)

    pts = [ps1, ps2, pe2, pe1]
    pts = np.array(pts).astype(int)

    img = cv2.drawContours(img, [pts],-1, 1, -1)

    return img

def getLines(frame, pt01, pt02, v_dir, r1, r2):

    height, width, _ = frame.shape

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.blur(gray_frame, (3,3))
    Gm = imgradientxy(gray_frame, v_dir)

    _minVal, _maxVal, _, _ = cv2.minMaxLoc(Gm)
    Gm -= _minVal
    Gm *= 255 / (_maxVal - _minVal)
    Gm = Gm.astype(np.uint8)

    Gm[np.where(Gm[:,:]<120)] = 0
    Gm[np.where(Gm[:,:]>0)] = 1

    rectimg = get_view_rect(pt01, pt02, v_dir, r1, r2, height, width)

    canny_img = cv2.Canny(gray_frame, 50, 200, None, 3)
    canny_img = canny_img.astype(np.uint8)
    canny_img = cv2.multiply(canny_img, Gm)
    canny_img = cv2.multiply(canny_img, rectimg)

    # show_frame = cv2.resize(canny_img, (600, (int)(600 * height / width)))
    # cv2.imshow("Canny Image", show_frame)
    # cv2.waitKey(1)




    dl = [pt02[0]-pt01[0], pt02[1]-pt01[1]]
    lpt12 = math.sqrt(dl[0]*dl[0] + dl[1] * dl[1])
    dl = [dl[0] / lpt12, dl[1] / lpt12]

    ptd = [pt02[0] - dl[0] * r2, pt02[1] - dl[1] * r2]



    theta0 = math.atan(dl[1]/dl[0]) + np.pi/2
    lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 60)
    # lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 100, None, 0, 0, min_theta=np.pi*95/180, max_theta=np.pi*177/180)
    # lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 100, None, 0, 0, min_theta=theta0-0.35, max_theta=theta0+0.35)

    cdst = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    cdst1 = cdst.copy()

    lines_ex = []

    if lines is not None:
        for i in range(0, len(lines)):
            # ll = lines[i]
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if (theta < theta0-0.35 or theta > theta0+0.35):
                continue
            cs = math.cos(theta)
            sn = math.sin(theta)
            x0 = cs * rho
            y0 = sn * rho

            x1, y1, x2, y2 = get2Point(x0, y0, -cs/sn, ptd[0], ptd[1], v_dir, height)

            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))

            d1 = is_near(pt1, pt2, pt01)
            d2 = is_near(pt1, pt2, pt02)
            # if abs(d1 - d2) > 0.6 * r2:
            #     continue

            if 0.5 * r1 < d1 and d1 < 2.2 * r1 and 0.5 * r2 < d2 and d2 < 2.2 * r2:
                lines_ex.append([pt1, pt2, d1, d2])
            # cv2.line(cdst1, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    # show_frame = cv2.resize(cdst1, (600, (int)(600 * height / width)))
    # cv2.imshow("Canny Image1", show_frame)
    # cv2.waitKey(1)
    ppp = 0

    # cv2.waitKey(0)
    canny_img = cv2.blur(canny_img, (3,3))
    scores1, scores2 = setLineScore(canny_img, Gm, lines_ex)

    coeff_gmscore = 0.0
    coeff_distscore = 20.0
    bese_sc = 0
    bese_idx = 0
    for i in range(len(lines_ex)):
        pt1, pt2, d1, d2 = lines_ex[i]
        if d1 < 1.3 * r1 and d2 < 1.3 * r2:
            sc1 = scores1[i]
            sc2 = scores2[i]
            sc3 = get_dist_score(d1, 0.7 * r1, d2, 0.7 * r2)
            sc = sc1 + coeff_gmscore * sc2 + coeff_distscore * sc3
            if bese_sc < sc :
                bese_sc = sc
                bese_idx = i
            # cdst1 = cdst.copy()
            # cv2.line(cdst1, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            # show_frame = cv2.resize(cdst1, (600, (int)(600 * height / width)))
            # cv2.imshow("Canny Image2", show_frame)
            # cv2.waitKey(1)
            ppp = 0


    basePt1, basePt2, baseD1, baseD2 = lines_ex[bese_idx]
    cv2.line(cdst, basePt1, basePt2, (0, 0, 255), 1, cv2.LINE_AA)


    bese_sc = 0
    bese_idx = 0
    for i in range(len(lines_ex)):
        pt1, pt2, d1, d2 = lines_ex[i]
        if baseD1 + 0.25 * r1 < d1 and baseD2 + 0.25 * r2 < d2:
            sc1 = scores1[i]
            sc2 = scores2[i]
            sc = sc1 + coeff_gmscore*sc2
            if bese_sc < sc:
                bese_sc = sc
                bese_idx = i
            # cdst1 = cdst.copy()
            # cv2.line(cdst1, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
            # show_frame = cv2.resize(cdst1, (600, (int)(600 * height / width)))
            # cv2.imshow("Canny Image3", show_frame)
            # cv2.waitKey(1)
            ppp = 0

    partPt1, partPt2, partD1, partD2 = lines_ex[bese_idx]

    # show_frame = cv2.resize(cdst, (600, (int)(600 * height / width)))
    # cv2.imshow("Canny Image4", show_frame)

    return [basePt1, basePt2, partPt1, partPt2]

def refineBoxes(boxes, height):
    nBoxes = len(boxes)
    minLeft = 1000
    maxLeft = 0
    minLeftIdx = -1
    maxLeftIdx = -1
    for i in range(nBoxes):
        box = boxes[i]
        left = box[1]
        if minLeft > left:
            minLeft = left
            minLeftIdx = i
        if maxLeft < left:
            maxLeft = left
            maxLeftIdx = i
    boxes1 = []
    boxes1.append(boxes[minLeftIdx])
    boxes1.append(boxes[maxLeftIdx])

    box0 = boxes1[0]
    left, top, right, bottom = box0[1], box0[0], box0[3], box0[2]

    if nBoxes == 3 and left == 0:
        minLeftIdx += 1
        box0 = boxes[minLeftIdx]
        left, top, right, bottom = box0[1], box0[0], box0[3], box0[2]

    if left < 2 and (right - left) < 0.85 * (bottom - top):
        left = right - 0.9 * (bottom - top)
    r1 = (right - left + bottom - top) / 4
    pt1 = [(right + left) / 2, (top + bottom) / 2]


    box0 = boxes1[1]
    left, top, right, bottom = box0[1], box0[0], box0[3], box0[2]

    if bottom > height-3 and (bottom - top) < 0.85 * (right - left):
        bottom = top + 0.9 * (right - left)
    r2 = (right - left + bottom - top) / 4
    pt2 = [(right + left) / 2, (top + bottom) / 2]

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dl = math.sqrt(dx*dx + dy*dy)
    dx /= dl
    dy /= dl

    v_dir = [dy, -dx]

    return [pt1, pt2, v_dir, r1, r2]
if __name__ == "__main__":
    import sys
