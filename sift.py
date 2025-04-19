import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Feature Matching and Object Detection using SIFT")
parser.add_argument('--query', required=True, help="Path to the query image (object)")
args = parser.parse_args()

# Load images
query_image_gray = cv2.imread(f'queries/{args.query}', cv2.IMREAD_GRAYSCALE)
query_image_color = cv2.imread(f'queries/{args.query}')

target_image_gray = cv2.imread('target.jpg', cv2.IMREAD_GRAYSCALE)
target_image_color = cv2.imread('target.jpg')

if query_image_gray is None or target_image_gray is None:
    print("Error loading images.")
    exit()

# SIFT detector
sift = cv2.SIFT_create()

query_keypoints, query_descriptors = sift.detectAndCompute(query_image_gray, None)
target_keypoints, target_descriptors = sift.detectAndCompute(target_image_gray, None)

# FLANN matcher 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann_matcher.knnMatch(query_descriptors, target_descriptors, k=2)

# Lowe's ratio test
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Homography
homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Rectangle around the detected object
h, w = query_image_gray.shape
object_points = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
transformed_points = cv2.perspectiveTransform(object_points, homography_matrix)

# Draw matches and detected rectangle
matches_image = cv2.drawMatches(query_image_color, query_keypoints, target_image_color, target_keypoints, good_matches, None, flags=2)
detected_image = cv2.polylines(target_image_color, [np.int32(transformed_points)], isClosed=True, color=(0, 255, 0), thickness=4)

# Convert to RGB for matplotlib    
matches_rgb = cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)
detected_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

# Save and display images
cv2.imwrite(f'outputs/matched-{args.query}', matches_image)
cv2.imwrite(f'outputs/detected-{args.query}', detected_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(matches_rgb)
plt.title('Feature Matches')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(detected_rgb)
plt.title('Detected Object')
plt.axis('off')

plt.show()
