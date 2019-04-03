import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
def calculate_error(path):
    chessboard_size = (9, 6)
    calibration_paths = glob.glob(path)
    obj_points = [] #3D points in real world space 
    img_points = [] #3D points in image plane
    #Prepare grid and points to display
    objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    #Iterate over images to find intrinsic matrix
    for image_path in tqdm(calibration_paths):
    #Load image
     image = cv2.imread(image_path)
     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     #find chessboard corners
     ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
    if ret == True:
      #define criteria for subpixel accuracy
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      #refine corner location (to subpixel accuracy) based on criteria.
      cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
      obj_points.append(objp)
      img_points.append(corners)
    #Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray_image.shape[::-1], None, None)
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    '''
    exif_img = PIL.Image.open(calibration_paths[0])
    exif_data = {
     PIL.ExifTags.TAGS[k]:v
     for k, v in exif_img._getexif().items()
     if k in PIL.ExifTags.TAGS}
    #Get focal length in tuple form
    focal_length_exif = exif_data['FocalLength']
    #Get focal length in decimal form
    focal_length = focal_length_exif[0]/focal_length_exif[1]
    '''
    INPUTARRAYMTX2 = mtx
    INPUTARRAYDISTCOEFFS2 = dist
    SIZE = (w,h)
    return (INPUTARRAYMTX2, INPUTARRAYDISTCOEFFS2, SIZE, obj_points, img_points)
