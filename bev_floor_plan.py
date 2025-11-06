import cv2
import numpy as np

def generate_floor_plan(bev_mask, save_path="output/bev/floor_plan.png"):
    bev = cv2.resize(bev_mask, (800, 800), interpolation=cv2.INTER_NEAREST)
    plan = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    plan[plan == 255] = 255
    plan[plan == 0] = 0
    cv2.imwrite(save_path, plan)
    return save_path
