# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored by Liyan Chen (liyanc@cs.utexas.edu)
#
import time
import tqdm
import torch
import matplotlib
import pygmtools as pygm

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from topologylayer.nn import WassersteinDistanceLayer
from matplotlib import pyplot as plt

plt.style.use('ggplot')
pygm.set_backend('pytorch')

torch.manual_seed(0)


pl1 = torch.randn(12, 128, 2, requires_grad=True)
pl2 = torch.randn(12, 72, 2, requires_grad=True)

pr1 = torch.randn(12, 160, 2, requires_grad=True) - 6.6
pr2 = torch.randn(12, 192, 2, requires_grad=True) + 3.6

losses = []
steps = 9116
init_lr, max_lr = 1e-4, 5e-3
pct_div = 8 / steps
opt = Adam([pl1, pl2], lr=2.5e-3)
wd = WassersteinDistanceLayer(norm="L2")
sch = OneCycleLR(
    opt, max_lr=max_lr, total_steps=steps, epochs=1, pct_start=pct_div, div_factor=max_lr / init_lr,
    final_div_factor=(6e-3/1e-6), base_momentum=0.05, max_momentum=0.97
)
for _ in (pb := tqdm.tqdm(range(steps))):
    c1, c2 = wd([pl1 * 3.2, pl2 * 1.1], [pr1, pr2])
    l = c1.mean() + c2.mean()
    losses.append((lf := float(l)))
    pb.set_postfix_str(f'Total cost: {lf:.6f}')
    l.backward()
    opt.step()
    sch.step()

fig, ax = plt.subplots(1, 1)
ax.plot(losses)
ax.set_title("Wasserstein Loss (Hungarian) between PD points\nof size (128 v. 160) + (96 v. 192)")
plt.savefig("../imgs/pd_loss.png")
