import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from calibrate_error import calculate_error
INPUTARRAYMTX_left, INPUTARRAYDISTCOEFFS_left, SIZE_L, objpoints_l, imgpoints_l, focal_length_left = calculate_error("left2.png", "left.png", "calibrated_left.png")
INPUTARRAYMTX_right, INPUTARRAYDISTCOEFFS_right, SIZE_R,objpoints_r, imgpoints_r, focal_length_right = calculate_error("right2.png", "right.png", "calibrated_right.png")
objpoints = objpoints_l
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, INPUTARRAYMTX_left, INPUTARRAYDISTCOEFFS_left, INPUTARRAYMTX_right, INPUTARRAYDISTCOEFFS_right, SIZE_L, None, None, None, None, cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap, leftROI, rightROI = cv2.stereoRectify(INPUTARRAYMTX_left, INPUTARRAYDISTCOEFFS_left, INPUTARRAYMTX_right, INPUTARRAYDISTCOEFFS_right, SIZE_R, R, T, None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, -1, SIZE_R)
leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        INPUTARRAYMTX_left, INPUTARRAYDISTCOEFFS_left, leftRectification,
        leftProjection, SIZE_L, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        INPUTARRAYMTX_right, INPUTARRAYDISTCOEFFS_right, rightRectification,
        rightProjection, SIZE_R, cv2.CV_32FC1)

np.savez_compressed("compressed", imageSize=SIZE_L,
        leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
        rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)
calibration = np.load("compressed.npz", allow_pickle=True)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

REMAP_INTERPOLATION = cv2.INTER_AREA
stereoMatcher = cv2.StereoBM_create(160,15)
stereoMatcher.setPreFilterSize(5)
stereoMatcher.setPreFilterCap(16)
stereoMatcher.setMinDisparity(-39)
stereoMatcher.setBlockSize(21)
stereoMatcher.setSpeckleRange(20)
stereoMatcher.setSpeckleWindowSize(0)
stereoMatcher.setTextureThreshold(507)
stereoMatcher.setUniquenessRatio(0)
cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(0)
while cap0.isOpened() and cap1.isOpened():
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0 = cv2.flip(frame0, -1)
    frame1 = cv2.flip(frame1, -1)
    leftFrame = frame0
    rightFrame = frame1

    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)
    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)
    DEPTH_VISUALIZATION_SCALE = 2048
    cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
