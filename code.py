# Template for Exercise 5 – Canny Edge Detector

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def gaussian_smoothing(img, sigma):
    """
    Apply Gaussian smoothing to reduce noise.
    """
    ksize = 5 
    gaussian_kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_kernel = gaussian_kernel_1d @ gaussian_kernel_1d.T
    smoothed_img = cv2.filter2D(img, -1, gaussian_kernel)
    
    return smoothed_img


def compute_gradients(img):
    """
    Compute gradient magnitude and direction (Sobel-based).
    Return gradient_magnitude, gradient_angle.
    """
    # Sobel kernels for x and y directions
    Gx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # Convert to float for magnitude
    gradient_magnitude = cv2.magnitude(Gx.astype(np.float64), Gy.astype(np.float64))

    # Compute gradient angle in radians
    gradient_angle = np.arctan2(Gy, Gx)

    return gradient_magnitude, gradient_angle


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.
    """
    H, W = mag.shape
    suppressed = np.zeros((H, W), dtype=np.float64)
    angle = ang * 180. / np.pi  # Converting radians to degrees
    angle[angle < 0] += 180     # Normalizing to [0, 180)

    # Quantizing angles to 4 main directions: 0, 45, 90, 135 degrees
    angle_quant = np.zeros_like(angle, dtype=np.uint8)
    angle_quant[(angle >= 0) & (angle < 22.5)] = 0
    angle_quant[(angle >= 157.5) & (angle <= 180)] = 0
    angle_quant[(angle >= 22.5) & (angle < 67.5)] = 45
    angle_quant[(angle >= 67.5) & (angle < 112.5)] = 90
    angle_quant[(angle >= 112.5) & (angle < 157.5)] = 135

     # Pad magnitude array to handle border pixels
    padded_mag = np.pad(mag, ((1, 1), (1, 1)), mode='constant')

     # Comparing pixel magnitude to neighbors in gradient directions
    for direction in [0, 45, 90, 135]:
        locs = np.where(angle_quant == direction)
        i = locs[0] + 1  # offset for padding
        j = locs[1] + 1

        if direction == 0:
            neighbors = np.stack([padded_mag[i, j+1], padded_mag[i, j-1]], axis=0)
        elif direction == 45:
            neighbors = np.stack([padded_mag[i+1, j-1], padded_mag[i-1, j+1]], axis=0)
        elif direction == 90:
            neighbors = np.stack([padded_mag[i+1, j], padded_mag[i-1, j]], axis=0)
        else:
            neighbors = np.stack([padded_mag[i-1, j-1], padded_mag[i+1, j+1]], axis=0)

        mag_vals = padded_mag[i, j]
        # Keeping value only if it's greater or equal to both neighbors
        mask = (mag_vals > neighbors[0]) & (mag_vals > neighbors[1])
        suppressed[locs[0][mask], locs[1][mask]] = mag_vals[mask]

    return suppressed


def double_threshold(nms, low, high):
    """
    Apply double thresholding to classify strong, weak, and non-edges.
    Return thresholded edge map.
    """
    strong = 2
    weak = 1
    thresholded = np.zeros_like(nms, dtype=np.uint8)

    # Strong edges: pixel magnitude >= high threshold
    thresholded[nms >= high] = strong #Pixels with gradient magnitude above the high threshold are classified as strong edges.

    # Weak edges: between low and high thresholds
    thresholded[(nms >= low) & (nms < high)] = weak #Pixels with magnitude between low and high thresholds are labeled as weak edges

    # Non-edges remain 0, Pixels below low threshold are suppressed as non-edges

    return thresholded


def hysteresis(edge_map, weak, strong):
    """
    Perform edge tracking by hysteresis.
    Return final binary edge map.
    """
    H, W = edge_map.shape
    final_edges = np.zeros((H, W), dtype=np.uint8)
    visited = np.zeros((H, W), dtype=bool)
    q = deque()

    for i in range(H):
        for j in range(W):
            if edge_map[i, j] == strong:
                final_edges[i, j] = 1
                visited[i, j] = True
                q.append((i, j))

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    while q:
        i, j = q.popleft()
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if not visited[ni, nj] and edge_map[ni, nj] == weak:
                    final_edges[ni, nj] = 1
                    visited[ni, nj] = True
                    q.append((ni, nj))

    return final_edges


def compute_metrics(manual_edges, cv_edges):
    """
    Compute MAD, precision, recall, and F1-score between two binary edge maps.
    """
    # Converting to boolean arrays for ease
    manual_bool = manual_edges.astype(bool)
    cv_bool = cv_edges.astype(bool)

    # Mean Absolute Difference (MAD)
    mad = np.mean(np.abs(manual_bool.astype(int) - cv_bool.astype(int)))

    # True positives (both edges present)
    tp = np.logical_and(manual_bool, cv_bool).sum()

    # False positives (manual has edge, opencv does not)
    fp = np.logical_and(manual_bool, np.logical_not(cv_bool)).sum()

    # False negatives (opencv has edge, manual does not)
    fn = np.logical_and(np.logical_not(manual_bool), cv_bool).sum()

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return mad, precision, recall, f1_score


# ==========================================================

# TODO: 1. Load the grayscale image 'bonn.jpg'
img = cv2.imread('data/bonn.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image 'bonn.jpg' not found or unable to load.")

# TODO: 2. Smooth the image using your Gaussian function
smoothed_img = gaussian_smoothing(img, sigma=0.27) #0.02

# TODO: 3. Compute gradients (magnitude and direction)
gradient_magnitude, gradient_angle = compute_gradients(smoothed_img)

# TODO: 4. Apply non-maximum suppression
nms_img = nonmax_suppression(gradient_magnitude, gradient_angle)

# TODO: 5. Apply double threshold (choose suitable low/high values)
low_threshold = 100
high_threshold = 240 
thresholded_img = double_threshold(nms_img, low_threshold, high_threshold)
print("Strong:", np.sum(thresholded_img == 2), "Weak:", np.sum(thresholded_img == 1))

# TODO: 6. Perform hysteresis to obtain final edges
final_edges = hysteresis(thresholded_img, weak=1, strong=2) 

# TODO: 7. Compare your result with cv2.Canny using MAD and F1-score
cv_edges = cv2.Canny(smoothed_img, low_threshold, high_threshold) // 255  # Normalize to 0/1

mad, precision, recall, f1_score = compute_metrics(final_edges, cv_edges)

print(f'MAD: {mad:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1_score:.4f}')


# TODO: 8. Display original image, your edges, and OpenCV edges
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1) # Original grayscale image
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2) # Your detected edges
plt.title('Your Canny Edges')
plt.imshow(final_edges, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3) # OpenCV detected edges 
plt.title('OpenCV Canny Edges')
plt.imshow(cv_edges / 255, cmap='gray')
plt.axis('off')

plt.show()

