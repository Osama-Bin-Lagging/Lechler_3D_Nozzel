import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import csv

# Load the image
#imp = int(input('img no'))
#for imp in range(imp, imp+1):
for imp in range(1, 17):
    image = cv2.imread('./Cropped_img/detected_green_'+str(imp)+'.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image

    kernel = np.ones((5, 5), np.uint8)

    # Apply erosion to remove small noise
    mask = cv2.erode(mask, kernel, iterations=2)

    # Apply dilation to restore the size of the object
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply closing to close small holes inside the foreground objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply opening to remove small objects from the foreground
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply Gaussian blur to the mask to further reduce noise
    #blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    res = cv2.bitwise_and(image, image, mask=mask)
    img_bw = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    each_cylinder_width_pix = img_bw.shape[1]//75
    circles = cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1.1, minDist=each_cylinder_width_pix*3//4, param1=5, param2=6.5, minRadius = 3, maxRadius = 15) 

    # Find contours in the mask
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_coordinates = []
        for i in circles[0, :]:
            center = [i[0],image.shape[1] - i[1]]
            circle_coordinates.append(center)
            # Draw the outer circle
            cv2.circle(img_bw, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img_bw, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Display the result
    #cv2.imshow('Detected Circles', img_bw)
    #cv2.waitKey(0)
    # Extract the centroid of each contour
    #centroids = []
    #for contour in contours:
    #    M = cv2.moments(contour)
    #    if M['m00'] != 0:
    #        cX = int(M['m10'] / M['m00'])
    #        cY = int(M['m01'] / M['m00'])
    #        centroids.append((cX, cY))

    # Convert list of centroids to numpy array for easier manipulation
    #centroids = np.array(centroids)

    # Separate the coordinates into x and y arrays
    #x = centroids[:, 0]
    #y = centroids[:, 1] 

    # Sort points by x-coordinate
    #sorted_indices = np.argsort(x)
    #x = x[sorted_indices]
    #y = y[sorted_indices]

    # Fit a univariate spline
    circle_coordinates.sort(key=lambda x: x[0])
    circle_coordinates = np.array(circle_coordinates)
    x, y = circle_coordinates[1:-1,0], np.array(circle_coordinates[1:-1,1])

    print(x.shape,y.shape)
    spline = UnivariateSpline(x, y, s= 3000)

    new_x = np.linspace(x[0], x[-1], 71)
    new_y = spline(new_x)
    print(y,new_x,new_y)
    with open('circle_coordinates.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_y)

    # Plotting the original image with detected green points and spline
    plt.figure(figsize=(8, 6))
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(x, y, color='green', label='Detected Points', s=50)  # Adjust size here (s=50)
    plt.scatter(new_x, new_y, color='orange', label='Saved Points', s=50)  # Adjust size here (s=50)
    plt.plot(x, spline(x), color='red', label='Univariate Spline')
    plt.title('Univariate Spline on Green Points')
    plt.legend()
    plt.axis('off')  # Turn off axis for better visualization
    #plt.show()

