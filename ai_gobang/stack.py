import numpy as np


class Stack:
    def __init__(self, capacity=4000, batch_size=8, channel=2, rotate=True):
        self.capacity = capacity
        self.observe = []
        self.action = []
        self.reward = []
        self.is_terminal = []
        self.batch_size = batch_size
        self.channel = channel
        self.rotate = rotate

    def insert(self, observe_t, action_t, reward_t, is_goon):
        self.observe.append(observe_t.astype(np.float))
        self.action.append(action_t)
        self.reward.append(reward_t)
        self.is_terminal.append(is_goon)

        if len(self.observe) > self.capacity:
            del (self.observe[0])
            del (self.action[0])
            del (self.reward[0])
            del (self.is_terminal[0])

    def get_batch(self):
        if len(self.observe)-1 <= self.channel - 1:
            return None
        idxes = np.random.randint(self.channel - 1, len(self.observe) - 1, self.batch_size)
        data = []
        for idx in idxes:
            ob_t = self.observe[idx + 1 - self.channel: idx + 1]  # t, w, h
            ob_t_ = self.observe[idx + 2 - self.channel: idx + 2]
            act = self.action[idx]   # 2
            reward = self.reward[idx]
            is_terminal = self.is_terminal[idx]
            data.append([ob_t, act, reward, ob_t_, is_terminal])
            if self.rotate:
                data.append([np.array(ob_t).transpose([0, 2, 1]).tolist(), [act[1], act[0]], reward,
                             np.array(ob_t_).transpose([0, 2, 1]).tolist(), is_terminal])

        return data

    def clean(self):
        self.observe = []
        self.action = []
        self.reward = []
        self.is_terminal = []


# stk = Stack(batch_size=3)
#
# stk.insert('o1', 'a1', 'r1')
# print(stk.get_batch())
#
# stk.insert('o2', 'a2', 'r2')
# print(stk.get_batch())
#
# stk.insert('o3', 'a3', 'r3')
# print(stk.get_batch())
#
# stk.insert('o4', 'a4', 'r4')
# print(stk.get_batch())
#
# stk.insert('o5', 'a5', 'r5')
# print(stk.get_batch())

