import numpy as np

class YoloUtils():
    def __init__(self, size, grid):
        self.size = size
        self.grid = grid # This is 7 in the paper.

        self.side = 1.0 / self.grid

    def get_grid_point(self, x, y):
        '''
        params:
            x: normalized x coordinate of a image (ex. 0.5)
            y: normalized y coordinate of a image (ex. 0.4)

        return:
            coordinates of a grid
        '''
        for i in range(self.grid):
            if x < self.side * (i+1): break
        for j in range(self.grid):
            if y < self.side * (j+1): break

        return (i, j)

    def make_class_map(self, labels):
        '''
        params:
            labels: a tuples [(class1, x1, y1, w1, h1), (class2, x2, y2, w2, h2), ... ]
        return:
            class_map: a numpy array.
        '''

        class_map = np.zeros([self.grid, self.grid], dtype='int32')
        class_map[:] = 20

        for label in labels:
            cl,x,y,w,h = label
            p1_x = x - w/2
            p1_y = y - h/2

            w_step = int(w // self.side)
            print(w_step)
            h_step = int(h // self.side)
            print(h_step)
            p1 = self.get_grid_point(p1_x, p1_y)
            p1 = self.convert(p1)
            class_map[p1] = cl
            for dw in range(w_step+1):
                for dh in range(h_step+1):
                    class_map[(p1[0]+dh, p1[1]+dw)] = cl
        return class_map

    def convert(self, xy):
        c, r = xy[1], xy[0]
        return (c,r)


utils = YoloUtils(480, 7)

print(utils.make_class_map([(18,0.546,0.5165165165165165,0.908,0.9669669669669669),
                            (14,0.145,0.6501501501501501,0.042,0.15915915915915915)]))
print(utils.make_class_map([(7,0.372,0.7794943820224719,0.48,0.44101123595505615),
                            (11,0.338,0.40308988764044945,0.24,0.4353932584269663)]))
print(utils.make_class_map([ (0,0.489,0.5813333333333334,0.922,0.3733333333333333) ]))
