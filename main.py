import cv2
from utils import detect_nickel_and_measure
from inference import inference
import ollama

NICKEL_AREA_MM2 = 350 

#img_pth = "datasets/test/true/food10.jpg"
img_pth = "food_nickel.jpg"
food_pixel_areas = inference(
    model_pth="runs/segment/trains/train41/weights/best.pt",
    img_pth=img_pth,
    display=False
)
#print(food_pixel_areas)

nickel_area_mm2 = detect_nickel_and_measure(img_pth, verbose=False) # consider segmentation model for coin detection instead of cv2

#print(int(nickel_area_mm2))

food_areas_mm = {}

for food, area in food_pixel_areas.items():
    food_areas_mm[food] = NICKEL_AREA_MM2 * (area / int(nickel_area_mm2))

#print(food_areas_mm)

client = ollama.Client("")

model = 'llama3'

#example = {'biscuit': 796.2928421543484, 'chicken duck': 2893.92632315225}

prompt = f"Here is a dictionary representing the square millimeters of each food: {food_areas_mm}. \
    Estimate the calories of each food item given its area. Output just a dictionary of the values."

# 1) convert area mm2 -> cm2
# 2) multiply by thickness to get cm3
# 3) estimate weight W = density x volume = in grams
# 4) convert grams to calories using reference calorie
out1 = client.generate(model, prompt)
print(out1.response)


# note: merge masks in inference so that overlapping masks arent counted again, which may overestimate area.
# play around with conf
# fix batch inference for debugging in inference.py
# fix inference so that it disregards overlapping masks for different classes. currently, it only disregards
# overlapping masks of same class.

# use bigger model for llm