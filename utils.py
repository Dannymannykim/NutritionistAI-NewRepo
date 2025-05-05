import torch
import math
import yaml
import numpy as np
import cv2

def compile_args(path):
    """
    Compiles and returns the arguments required for the external, process configuration.

    Args:
        path (str): 
            The .yaml path to the configuration or environment data.

    Returns:
        dict: external_args, train_args, tune_args, hyperparameters
    """

    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    external_args = config['external']
    train_args = config['train']
    tune_args = config.get('tune', {})
    hyperparameters = config.get('hyperparameters', {})

    return external_args, train_args, tune_args, hyperparameters

def detect_nickel_and_measure(img_pth, verbose=False):
    """
    Uses openCV2 to detect and mask a coin, and outputs
    the pixel area. Note that 

    Args:
        img (str):
            Path to image file.
        verbose (bool):
            Set to True to see mask overlay for debugging, with
            hsv and percent_silver values.
    
    Returns:
        pixel_area (int):
            Area of nickel in pixels.   
    """
    if verbose:
        cv2.namedWindow('My Resizable Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('My Resizable Window', 600, 400)

    image = cv2.imread(img_pth)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    h, w, c = image.shape 
    #print(h,w,c, 'coin')
    #image = cv2.imread('datasets/test/true/food11.jpg')
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=280,  # Canny high threshold
        param2=55,  # lower than usual → accept partial circles
        minRadius=20, maxRadius=600
    )

    # hue = base color, saturation = tells how pure (high) or grayish (low) color is, value = bright (255) or dark (0)
    # for nickel, saturation < 60-70 seems best
    coin_count = 0
    pixel_area = 0
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circlesRound = np.round(circles[0, :]).astype("int")
        
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circlesRound:
            if coin_count > 1:
                raise ValueError("More than one coin detected!")
            #cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            
            mask = np.zeros(blur.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            mean_hsv = cv2.mean(hsv, mask=mask)[:3]  # (H, S, V)
            h, s, v = mean_hsv

            silver_pixels = cv2.inRange(hsv, (0, 0, 120), (179, 80, 255))
            silver_count = cv2.countNonZero(cv2.bitwise_and(silver_pixels, silver_pixels, mask=mask))
            total_count = cv2.countNonZero(mask)
            percent_silver = silver_count / total_count * 100
            
            if verbose:
                print(f"h: {h}, s: {s}, v: {v}, percent_silver: {percent_silver}")

            if s < 65 and 80 < v < 200 and percent_silver >= 70:
                color = (0, 255, 0)  # for visual purposes: Green = likely nickel
                coin_count += 1
                #print(x,y,r,'sds')
                pixel_area = np.pi * r**2
            else:
                color = (0, 0, 255)  # for visual purposes: Red = reject

            cv2.circle(output, (x, y), r, color, 4)
        
        if verbose:
            cv2.imshow('My Resizable Window', output)
            cv2.waitKey(0)
        return pixel_area
    else:
        print('No circles found')

    
if __name__ == '__main__':

    # testing nickel detection
    img = "nickel.jpeg"
    pixel_area = detect_nickel_and_measure(img, verbose=True)
    print(pixel_area)
