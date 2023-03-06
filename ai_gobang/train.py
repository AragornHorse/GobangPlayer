import numpy as np
import torch
from goband import Gobang
from ai import AI
from stack import Stack


map_size = [12, 12]
attn_time = 4
max_time = 100

gb = Gobang(win_len=5, map_size=map_size)
ai = AI(map_size, attn_time, device=torch.device("cuda"), gamma=0.9, lr=2e-5, step_size=1000)
stk = Stack(capacity=4000, batch_size=4, channel=attn_time, rotate=True)

for epoch in range(10000):
    for _ in range(10):
        i, j = ai.train_policy(np.zeros([attn_time, map_size[0], map_size[1]]))
        r, goon, ob = gb.add_chess([i, j])
        stk.insert(reward_t=r, observe_t=ob, is_goon=goon, action_t=[i, j])

        i, j = ai.enemy_eval(np.zeros([attn_time, map_size[0], map_size[1]]), gb.map)
        gb.add_chess([i, j])

    for _ in range(100):
        data = stk.get_batch()
        loss = ai.train(data)
        print(loss)

    stk.clean()
    ai.update_assistant()























