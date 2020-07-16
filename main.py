from __future__ import print_function
from cv2 import cv2
import numpy as np

video_path = 'video/cropped_model2.mp4'
video = cv2.VideoCapture(video_path)

linux = cv2.imread('images/logo_2.jpg')
linux = cv2.resize(linux, (30, 30))

# padding to needed place
linux = cv2.copyMakeBorder(linux, 90, 168, 90, 110, borderType=cv2.BORDER_CONSTANT)

person = None
old_person = None
linux_transformed = None

MAX_FEATURES = 3000
GOOD_MATCH_PERCENT = 0.01

def alignImages(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    if len(matches) < 4:
        return None, None

    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)


    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = im2.shape

    if type(h) != np.ndarray:

        return None, None

    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def video_loop():
    i = 0
    while True:
        ret, frame = video.read()

        if i == 0:
            cv2.imwrite('images/ideal.jpg', frame)

        if not ret:
            break

        person = frame #detection
        if old_person is not None:
            imReg, h = alignImages(person, old_person)

            if type(h) == np.ndarray:

                linux_transformed = cv2.warpPerspective(
                                        linux, h,
                                        (frame.shape[1], frame.shape[0]),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0, 0))

                added_image = cv2.addWeighted(frame, 1, linux_transformed, 1, 0)

            else:

                added_image = frame
            
        else:
            if type(linux_transformed) == np.ndarray:
                added_image = cv2.addWeighted(frame, 1, linux_transformed, 1, 1)
            else:
                added_image = frame

        cv2.imshow("img", added_image)
        ch = 0xFF & cv2.waitKey(50)
        if ch == 27:
            break
        
        old_person = person

        if type(linux_transformed) == np.ndarray:
            linux = linux_transformed
        i += 1

    video.release()
    cv2.destroyAllWindows()

#ideal = cv2.imread('images/ideal.jpg')
#linux = cv2.imread('images/logo_2.jpg')
#linux = cv2.resize(linux, (30, 30))

#linux = cv2.copyMakeBorder(linux, 90, 168, 90, 110, borderType=cv2.BORDER_CONSTANT)

#added_image = cv2.addWeighted(ideal, 1, linux, 1, 0)
#cv2.imwrite('images/ideal_linux.jpg', added_image)