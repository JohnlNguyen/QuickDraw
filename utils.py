#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import os
import pandas as pd

BASE_DIR = os.getcwd()


# In[23]:


from PIL import Image, ImageDraw

def draw_it(raw_strokes):
    image = Image.new("P", (255,255), color=255)
    image_draw = ImageDraw.Draw(image)

    for stroke in eval(raw_strokes):
        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                            fill=0, width=6)
    return image


# In[24]:

def test():
    draw_type = "wheel"
    out_dir = f'{BASE_DIR}/images/{draw_type}'
    data = pd.read_csv(f'{BASE_DIR}/train/{draw_type}.csv')
    draw_it(data.loc[0].drawing)

if __name__ == '__main__':
    test()
# In[25]:




