


from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np

def fudge_luminance(colors, m, b=0):
    hsv = rgb_to_hsv(colors)
    hsv[:,2]*=m
    hsv[:,2]+=b
    return hsv_to_rgb(hsv.clip(0,1))


emotion_labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

emotion_label_colors = {
    'angry': '#FF0000',  # Red
    'disgust': '#808000', # Olive/Dark Yellow-Green
    'fear': '#800080',    # Purple
    'happy': '#FFFF00',   # Yellow
    'sad': '#0000FF',     # Blue
    'surprise': '#FFA500', # Orange
    'neutral': '#808080'  # Gray
}
