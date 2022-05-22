# this is manual procedure to estimate the warping from camera view to google-earth view
import cv2
import numpy as np


def find_key_points(im_src, im_dst):
    while True:
        cv2.imshow('source', im_src)
        cv2.imshow('destination', im_dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    im_src = cv2.imread('Target_street_Full.png')
    im_dst = cv2.imread('Template_marked.jpg')
    find_key_points(im_src, im_dst)
    # once manually written down the key poiny coordinates of the two images, then enter them in these two matrices

    # Destination (google-earth)
    Destin_r = [495, 72]
    Destin_g = [635, 435]
    Destin_b = [154, 229]
    Destin_c = [300, 466]

    # source image (camera view)
    Target_r = [199, 550]
    Target_g = [1000, 228]
    Target_b = [1638, 1072]
    Target_c = [1660, 372]

    pts_src = np.array([Target_r, Target_g, Target_b, Target_c, Target_c])
    # pts_src = np.array([Target_r, Target_g, Target_b, Target_c])
    pts_dst = np.array([Destin_r, Destin_g, Destin_b, Destin_c, Destin_c])

    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
    cv2.imwrite('RotStreet.png', im_out)

    np.savetxt('matrix.txt', h)

