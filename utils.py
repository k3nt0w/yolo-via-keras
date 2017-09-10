import numpy as np

class YoloUtils():
    def __init__(self, size=480, grid=7, nb_class=20):
        self.size = size
        self.grid = grid
        self.nb_class = nb_class

        self.side = 1.0 / self.grid

    def make_train_map(self, label):
        '''
        params:
            labels: a tuples (class1, x1, y1, w1, h1)
        return:
            class_map: a numpy array.
        '''

        class_map = np.zeros([self.grid, self.grid], dtype='int32')
        class_map[:] = 20

        cl,x,y,w,h = label

        object_map = np.zeros([10, self.grid, self.grid])
        gp = self._convert(self._get_grid_point(x, y))
        object_map[:, gp[0], gp[1]] = (label[1:] + [1,]) * 2

        p1_x = x - w/2
        p1_y = y - h/2
        w_step = int(w // self.side)
        h_step = int(h // self.side)
        p1 = self._get_grid_point(p1_x, p1_y)
        p1 = self._convert(p1)
        class_map[p1] = cl
        for dw in range(w_step+1):
            for dh in range(h_step+1):
                class_map[(p1[0]+dh, p1[1]+dw)] = cl
        class_map = self._binarylab(class_map)

        return np.concatenate([object_map,class_map])

    def _get_grid_point(self, x, y):
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

    def _binarylab(self, class_map):
        y = np.zeros((self.nb_class,self.grid,self.grid))
        for i in range(self.grid):
            for j in range(self.grid):
                cl = class_map[i][j]
                if cl == self.nb_class: continue
                y[cl, i, j] = 1
        return y

    def _convert(self, xy):
        c, r = xy[1], xy[0]
        return (c,r)

if __name__ == "__main__":
    utils = YoloUtils(480, 7)

    print(utils.make_class_map([(18,0.546,0.5165165165165165,0.908,0.9669669669669669),
                                (14,0.145,0.6501501501501501,0.042,0.15915915915915915)]))
    print(utils.make_class_map([(7,0.372,0.7794943820224719,0.48,0.44101123595505615),
                                (11,0.338,0.40308988764044945,0.24,0.4353932584269663)]))
    print(utils.make_class_map([ (0,0.489,0.5813333333333334,0.922,0.3733333333333333) ]))
