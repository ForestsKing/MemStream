import argparse
import numpy as np
import scipy
from sklearn import metrics

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


class MemStream(nn.Module):
    def __init__(self, params):
        super(MemStream, self).__init__()
        self.in_dim = params['in_dim']
        self.out_dim = params['out_dim']
        self.device = params['device']
        self.memory_len = params['memory_len']
        self.max_thres = params['beta']
        self.K = params['k']
        self.exp = torch.Tensor([params['gamma']**i for i in range(self.K)]).to(self.device)
        self.count = 0
        self.mean = None
        self.std = None

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim)
        ).to(self.device)

        # 存放原始数据，以便获得均值方差
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(self.device)
        self.mem_data.requires_grad = False
        # 存放编码，真正的记忆模块
        self.memory = torch.randn(self.memory_len, self.out_dim).to(self.device)
        self.memory.requires_grad = False

    def train_autoencoder(self, data, epochs, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        mean, std = data.mean(dim=0), data.std(dim=0)
        data = (data - mean) / std
        data[:, std == 0] = 0
        data = Variable(data)

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            output = self.decoder(self.encoder(data + 0.001*torch.randn_like(data).to(self.device)))
            loss = self.loss_fn(output, data)
            loss.backward()
            self.optimizer.step()

    def initialize_memory(self, data):
        resid = data.clone()

        mean, std = data.mean(dim=0), data.std(dim=0)
        data = (data - mean) / std
        data[:, std == 0] = 0
        data = Variable(data)

        self.memory = self.encoder(data)
        self.memory.requires_grad = False

        self.mem_data = resid
        self.mem_data.requires_grad = False
        self.mean, self.std = self.mem_data.mean(dim=0), self.mem_data.std(dim=0)

    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count % self.memory_len
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(dim=0), self.mem_data.std(dim=0)
            self.count += 1
            return 1
        return 0

    def forward(self, data):
        resid = data.clone()

        data = (data - self.mean) / self.std
        data[:, self.std == 0] = 0
        encoder_output = self.encoder(data)

        loss_values = (torch.topk(torch.norm(self.memory - encoder_output, dim=1, p=1), k=self.K, largest=False)[0]*self.exp).sum()/self.exp.sum()

        self.update_memory(loss_values, encoder_output, resid)
        return loss_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pima')

    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument("--gamma", type=float, help="knn coefficient", default=0)
    parser.add_argument("--memlen", type=int, help="size of memory", default=64)
    parser.add_argument("--k", type=int, default=3)

    parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)

    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--dev", help="device", default="cuda:1")

    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)

    # 设置 CPU 或者 GPU
    device = torch.device(args.dev)

    # 加载数据
    N = args.memlen
    df = scipy.io.loadmat('./dataset/'+args.dataset+".mat")
    numeric = torch.FloatTensor(df['X'])
    labels = (df['y']).astype(float).reshape(-1)
    data_loader = DataLoader(numeric, batch_size=1)
    init_data = numeric[labels == 0][:N].to(device)
    print(numeric.shape, labels.shape)

    # 定义模型
    params = {
        'in_dim': numeric.shape[1],
        'out_dim': int(1 * numeric.shape[1]),
        'device': device,
        'memory_len': N,
        'beta': args.beta,
        'gamma': args.gamma,
        'k': args.k,
    }
    model = MemStream(params).to(device)

    # 训练加噪自编码器
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs, lr=args.lr)

    # 初始化记忆体
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data).to(device))

    # 流式检测
    err = []
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    scores = np.array([i.cpu() for i in err])
    auc = metrics.roc_auc_score(labels, scores)
    print("ROC-AUC", auc)
