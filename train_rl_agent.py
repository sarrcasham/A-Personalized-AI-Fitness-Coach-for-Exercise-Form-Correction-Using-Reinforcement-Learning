import numpy as np

def normalize_biometrics(age, height, weight, gender):
    norm_age = (age - 10) / (80 - 10)
    norm_height = (height - 140) / (220 - 140)
    norm_weight = (weight - 40) / (150 - 40)
    norm_gender = float(gender)
    
    norm_age = np.clip(norm_age, 0, 1)
    norm_height = np.clip(norm_height, 0, 1)
    norm_weight = np.clip(norm_weight, 0, 1)
    
    return np.array([norm_age, norm_height, norm_weight, norm_gender], dtype=np.float32)

def denormalize_biometrics(norm_array):
    norm_age, norm_height, norm_weight, norm_gender = norm_array
    
    age = norm_age * (80 - 10) + 10
    height = norm_height * (220 - 140) + 140
    weight = norm_weight * (150 - 40) + 40
    gender = int(norm_gender)
    
    return age, height, weight, gender
