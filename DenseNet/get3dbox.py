'''
For MRI/CT version of the model if using 2D RPN on each slice, and then projecting onto
3D space via greedy max.

TODO:
    1. Come up with how no bbox outputted for a given layer will be represnted. w,h=0 or -1 or something else?
    2. Complete unnormalize function. Depends on how we normalize the outputs.
    3. Get new 3D RoI tensor to be fed into segmentation network.

'''

def unnormalize(x):
    return x

def get3dbox(positions):
    '''
    positions contains tuples (x, y, w, h, d) where

    len(positions) -> D, total depth of volume
    x, y are coordinates of bbox center for slice at depth d.
    w and h are the height of the bbox (after normalized, have to process it but assume its processed already here)
    Note: x,y,w,h are normalized as these are outputs of the RoI, and d is integer, the depth index of a given slice.

    p0 ----- p1
    |        |
    |        |
    p2 ----- p3

    '''

    x_0 = None
    x_1 = None
    x_2 = None
    x_3 = None

    y_0 = None
    y_1 = None
    y_2 = None
    y_3 = None

    d_min = None
    d_max = None

    levels = len(positions)
    for d in range(levels):
        if w == 0 or h == 0: #or == -1 for no bbox, gotta figure out how no bbox outputted is represented. (!!!)
            continue

        if d_min == None:
            d_min = d

        d_max = d

        x = unnormalize(positions[d][0])
        y = unnormalize(positions[d][1])
        w = unnormalize(positions[d][2])
        h = unnormalize(positions[d][3])

        #p0
        x_0 = x-w/2 if x-w/2 < x_0 else x_0
        y_0 = y-h/2 if y-h/2 < y_0 else y_0

        #p1
        x_1 = x+w/2 if x+w/2 > x_1 else x_1
        y_1 = y-h/2 if y-h/2 < y_1 else y_1

        #p2
        x_2 = x-w/2 if x-w/2 < x_2 else x_2
        y_2 = y+h/2 if y+h/2 > y_2 else y_2

        #p3
        x_3 = x+w/2 if x+w/2 > x_3 else x_3
        y_3 = y+w/2 if y+w/2 > y_3 else y_3

    x_left = min(x_0, x_2)
    x_right = max(x_1, x_3)
    y_top = min(y_0, y_1)
    y_bottom = min(y_2, y_3)

    return x_left, x_right, y_top, y_bottom, d_min, d_max
