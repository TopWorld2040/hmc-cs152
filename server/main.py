import os

ITER = 15
CONTENT_WEIGHT = [1]
STYLE_WEIGHT = [100,1000,10000]
VARIATION_WEIGHT = [1, 10, 100]

PYTHON = 'python'
FILENAME = 'image_style.py'
BASE = 'Tuebingen_Neckarfront.jpg'
STYLE = '100iteration.jpg'

for content in CONTENT_WEIGHT:
    for style in STYLE_WEIGHT:
        for variation in VARIATION_WEIGHT:
            OUTPUT = '[cw={},sw={},vw={}]'.format(content,style,variation)
            os.system('{} {} --iter {} --content_weight {} --style_weight {} --tv_weight {} {} {} {}'.format(PYTHON, FILENAME, ITER, content, style, variation, BASE, STYLE, OUTPUT))


