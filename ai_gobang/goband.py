import torch
import numpy as np



class Gobang:
    def __init__(self, win_len=5, map_size=None):
        if map_size is None:
            map_size = [12, 12]
        self.map_size = map_size
        self.win_len = win_len
        self.size = map_size
        self.map = np.zeros(map_size)
        self.black = 1
        self.white = -1
        self.cur_player = 1
        self.time = 0

    def __str__(self):
        m = "  "
        for i in range(self.size[1]):
            m += "{} ".format(i)
        m += '\n'
        for i, x in enumerate(self.map):
            m += '{} '.format(i)
            for p in x:
                if p == 0:
                    m += '  '
                elif p == 1:
                    m += 'x '
                else:
                    m += 'o '
            m += '\n'
        return m

    def get_same_num(self, pos, dire='up'):  # ij
        if dire == 'up':
            num = 0
            clor = 0
            i, j = pos
            i = i - 1
            if i >= 0:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while i >= 0 and self.map[i, j] == clor:
                num += 1
                i = i - 1
            return num, clor
        elif dire == 'down':
            num = 0
            clor = 0
            i, j = pos
            i = i + 1
            if i < self.size[0]:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while i < self.size[0] and self.map[i, j] == clor:
                num += 1
                i = i + 1
            return num, clor
        elif dire == 'right':
            num = 0
            clor = 0
            i, j = pos
            j = j + 1
            if j < self.size[1]:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while j < self.size[1] and self.map[i, j] == clor:
                num += 1
                j = j + 1
            return num, clor
        elif dire == 'left':
            num = 0
            clor = 0
            i, j = pos
            j = j - 1
            if j >= 0:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while j >= 0 and self.map[i, j] == clor:
                num += 1
                j = j - 1
            return num, clor
        elif dire == 'left_up':
            num = 0
            clor = 0
            i, j = pos
            j = j - 1
            i = i - 1
            if j >= 0 and i >= 0:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while j >= 0 and i >= 0 and self.map[i, j] == clor:
                num += 1
                j = j - 1
                i = i - 1
            return num, clor
        elif dire == 'left_down':
            num = 0
            clor = 0
            i, j = pos
            j = j - 1
            i = i + 1
            if j >= 0 and i < self.size[0]:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while j >= 0 and i < self.size[0] and self.map[i, j] == clor:
                num += 1
                j = j - 1
                i = i + 1
            return num, clor
        elif dire == 'right_down':
            num = 0
            clor = 0
            i, j = pos
            j = j + 1
            i = i + 1
            if j < self.size[1] and i < self.size[0]:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while j < self.size[1] and i < self.size[0] and self.map[i, j] == clor:
                num += 1
                j = j + 1
                i = i + 1
            return num, clor
        elif dire == 'right_up':
            num = 0
            clor = 0
            i, j = pos
            j = j + 1
            i = i - 1
            if j < self.size[1] and i >= 0:
                clor = self.map[i, j]
            if clor == 0:
                return 0, 0  # num, color
            while j < self.size[1] and i >= 0 and self.map[i, j] == clor:
                num += 1
                j = j + 1
                i = i - 1
            return num, clor

    def add_chess(self, pos, mode="ij"):
        ob = self.map
        self.time += 1
        j, i = pos
        if mode == 'ij':
            i, j = [j, i]
        if self.map[i, j] != 0:
            self.cur_player *= -1
            return -10000, True, ob

        self.map[i, j] = self.cur_player
        reward = 0
        ctn = True

        up_num, up_clor = self.get_same_num([i, j], 'up')
        down_num, down_clor = self.get_same_num([i, j], 'down')
        right_num, right_clor = self.get_same_num([i, j], 'right')
        left_num, left_clor = self.get_same_num([i, j], 'left')
        # print("up", up_num, up_clor)
        # print("down", down_num, down_clor)
        # print("left", left_num, left_clor)
        # print("right", right_num, right_clor)

        if up_num >= self.win_len - 1:
            if up_clor == self.cur_player:
                # print("171")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if down_num >= self.win_len - 1:
            if down_clor == self.cur_player:
                # print("180")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if left_num >= self.win_len - 1:
            if left_clor == self.cur_player:
                # print("190")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if right_num >= self.win_len - 1:
            if right_clor == self.cur_player:
                # print("198")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if up_clor == down_clor:
            if up_num + down_num >= self.win_len - 1:
                if up_clor == self.cur_player:
                    # print("208")
                    self.clean()
                    return 10000, False, ob
                else:
                    self.cur_player *= -1
                    return 5000, True, ob

        if left_clor == right_clor:
            if left_num + right_num >= self.win_len - 1:
                if right_clor == self.cur_player:
                    # print("218")
                    self.clean()
                    return 10000, False, ob
                else:
                    self.cur_player *= -1
                    return 5000, True, ob

        reward += up_num
        reward += down_num
        reward += left_num
        reward += right_num

        up_num, up_clor = self.get_same_num([i, j], 'right_up')
        down_num, down_clor = self.get_same_num([i, j], 'left_down')
        right_num, right_clor = self.get_same_num([i, j], 'right_down')
        left_num, left_clor = self.get_same_num([i, j], 'left_up')

        if up_num >= self.win_len - 1:
            if up_clor == self.cur_player:
                # print("237")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if down_num >= self.win_len - 1:
            if down_clor == self.cur_player:
                # print("246")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if left_num >= self.win_len - 1:
            if left_clor == self.cur_player:
                # print("255")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if right_num >= self.win_len - 1:
            if right_clor == self.cur_player:
                # print("264")
                self.clean()
                return 10000, False, ob
            else:
                self.cur_player *= -1
                return 5000, True, ob

        if up_clor == down_clor:
            if up_num + down_num >= self.win_len - 1:
                if up_clor == self.cur_player:
                    # print("274")
                    self.clean()
                    return 10000, False, ob
                else:
                    self.cur_player *= -1
                    return 5000, True, ob

        if left_clor == right_clor:
            if left_num + right_num >= self.win_len - 1:
                if right_clor == self.cur_player:
                    # print("284")
                    self.clean()
                    return 10000, False, ob
                else:
                    self.cur_player *= -1
                    return 5000, True, ob

        reward += up_num
        reward += down_num
        reward += left_num
        reward += right_num

        self.cur_player *= -1
        return reward, ctn, ob

    def get_map(self, device=torch.device("cpu")):
        return torch.tensor(self.map, device=device)

    def clean(self):
        # print(self.__str__())
        self.map = np.zeros(self.map_size)
        self.time = 0
        self.cur_player = self.black

# g = Gobang()
#
# while True:
#     try:
#         i, j = input("ij>>>").split(',')
#         i = int(i)
#         j = int(j)
#         r, c = g.add_chess([i, j], mode='ij')
#         print(g)
#         print(r, c)
#     except:
#         pass











