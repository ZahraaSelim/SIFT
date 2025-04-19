import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Feature Matching and Object Detection using SIFT")
parser.add_argument('--query', required=True, help="Path to the query image (object)")
parser.add_argument('--target', required=True, help="Path to the target image (scene)")
parser.add_argument('--output', required=True, help="Path to save the output image")
args = parser.parse_args()

# Load images
query_img = cv2.imread(args.query, cv2.IMREAD_GRAYSCALE)
target_img = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)
target_color = cv2.imread(args.target)

if query_img is None or target_img is None:
    print("Error loading images.")
    exit()

# SIFT detector
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(query_img, None)
kp2, des2 = sift.detectAndCompute(target_img, None)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = query_img.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Draw rectangle
    detected = cv2.polylines(target_color, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=4)

    # Resize for display
    display_width = 1000
    scale = display_width / detected.shape[1]
    display_height = int(detected.shape[0] * scale)
    resized = cv2.resize(detected, (display_width, display_height))

    # Save and plot
    cv2.imwrite(args.output, detected)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title('Detected Object')
    plt.show()

else:
    print("Not enough good matches were found.")
