import time
import torch
from torch import nn
from torch.nn.parameter import Parameter
from model.dualcracknet import DualCrackNet
def fps_params_flops(model, size):
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(size).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS, ">>>res. ", size)

if __name__ == "__main__":
    model = DualCrackNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval
    fps_params_flops(model, (1, 3, 320, 320))