import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


class ai(nn.Module):
    def __init__(self, in_size=None, in_channel=2):
        super(ai, self).__init__()
        if in_size is None:
            in_size = [12, 12]
        self.in_size = in_size
        self.in_channel = in_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=8, stride=1, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.tsconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=1, padding=0, bias=False)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=4, stride=2, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(4, 1, stride=2, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.tsconv(out)
        short = self.shortcut(x)
        return out.squeeze() + short.squeeze()


class AI:
    def __init__(self, in_size=None, in_channel=2, device=torch.device("cuda"), gamma=0.9, loss_func=None,
                 lr=1e-3, step_size=1000):
        if in_size is None:
            in_size = [12, 12]
        self.in_size = in_size
        self.ai = ai(in_size, in_channel).to(device)
        self.assistant = ai(in_size, in_channel).to(device)
        self.device = device
        self.gamma = gamma
        self.loss_func = loss_func if loss_func is not None else nn.MSELoss()
        self.opt = optim.Adam(self.ai.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size, 0.9)

    def load(self, path=None):
        if path is None:
            path = r"./parameter.pth"
        self.assistant.load_state_dict(torch.load(path))
        self.ai.load_state_dict(torch.load(path))

    def update_assistant(self, path=None):
        if path is None:
            path = r"./parameter.pth"
        torch.save(self.ai.state_dict(), path)
        self.assistant.load_state_dict(torch.load(path))

    def train(self, data):  # o_t, a_t, r_t, o_t_, is_goon
        if data is None:
            return None
        o_t = []
        a_t = []
        r_t = []
        o_t_ = []
        is_goon = []
        for d in data:
            o, a, r, o_, i = d
            o_t.append(o)
            a_t.append(a)
            r_t.append(r)
            o_t_.append(o_)
            is_goon.append(i)

        o_t = torch.tensor(np.array(o_t), device=self.device, dtype=torch.float)   # b, c, w, h
        a_t = np.array(a_t).transpose([1, 0]).tolist()   # b, 2
        r_t = torch.tensor(r_t, device=self.device, dtype=torch.float)   # b
        o_t_ = torch.tensor(np.array(o_t_), device=self.device, dtype=torch.float)  # b, c, w, h
        is_goon = torch.tensor(is_goon, device=self.device) + 0  # b

        qs_t = self.ai(o_t)  # b, w, h
        qs_t_ = self.assistant(o_t_).detach()  # b, w, h

        q_t = qs_t[range(0, qs_t.size(0)), a_t[0], a_t[1]]
        # q_t = []
        # for i, idx in enumerate(a_t):
        #     q_t.append(qs_t[i, idx[0], idx[1]])
        #
        # q_t = torch.tensor(q_t, device=self.device)  # b

        q_t_, _ = torch.max(qs_t_.view(o_t.size(0), -1), dim=-1)  # b

        q_t_ = q_t_ * is_goon * self.gamma + r_t  # b

        # print(q_t_.size())
        # print(q_t.size())

        loss = self.loss_func(q_t, q_t_)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        return loss

    def train_policy(self, ob, p_greedy=0.5):
        if random.random() < p_greedy:
            ob = torch.tensor(ob, device=self.device, dtype=torch.float).unsqueeze(0)  # 1, c, w, h
            qs = self.ai(ob).squeeze().detach().cpu()  # w, h
            i_best, i = torch.max(qs, dim=0)  # h
            j = torch.argmax(i_best)
            i = i[j]
        else:
            i = random.randint(0, self.in_size[0]-1)
            j = random.randint(0, self.in_size[1]-1)
        # print(i, j)
        return i, j

    def eval(self, ob, now):
        can = (torch.tensor(now, device=self.device, dtype=torch.float).squeeze() == 0) + 0  # w, h
        qs = (self.ai(torch.tensor(ob, device=self.device, dtype=torch.float).unsqueeze(0)) * can).detach().cpu()
        i_best, i = torch.max(qs, dim=0)  # h
        j = torch.argmax(i_best)
        i = i[j]
        return i, j

    def enemy_eval(self, ob, now):
        can = (torch.tensor(now, device=self.device, dtype=torch.float).squeeze() == 0) + 0  # w, h
        qs = (self.assistant(-torch.tensor(ob, device=self.device, dtype=torch.float).unsqueeze(0)) * can).detach().cpu()
        i_best, i = torch.max(qs, dim=0)  # h
        j = torch.argmax(i_best)
        i = i[j]
        return i, j







# x = torch.ones(8, 2, 12, 12)
# md = ai()
# out = md(x)
# print(out.size())


