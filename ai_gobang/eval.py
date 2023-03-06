import numpy as np

from ai import AI
from goband import Gobang
import torch

map_size = [12, 12]
attn_time = 4
max_time = 100

gb = Gobang(win_len=5, map_size=map_size)
ai = AI(map_size, attn_time, device=torch.device("cuda"), gamma=0.9, lr=1e-4, step_size=1000)
ai.load()
ob = np.zeros([attn_time, map_size[0], map_size[1]])

i, j = ai.eval(ob, gb.map)
# print(i,j)
r, goon, o = gb.add_chess([i, j])
ob[:-1, :, :] = ob[1:, :, :]
ob[-1, :, :] = o
print(gb)

while True:
    try:
        i, j = input(">>>").split(',')
        i = int(i)
        j = int(j)
    except:
        continue
    gb.add_chess([i, j])

    i, j = ai.eval(ob, gb.map)
    print(i,j)
    r, goon, o = gb.add_chess([i, j])
    ob[:-1, :, :] = ob[1:, :, :]
    ob[-1, :, :] = o
    if not goon:
        i, j = ai.eval(ob, gb.map)
        # print(i,j)
        r, goon, o = gb.add_chess([i, j])
        ob[:-1, :, :] = ob[1:, :, :]
        ob[-1, :, :] = o
        print(gb)
        continue
    print(gb)





















