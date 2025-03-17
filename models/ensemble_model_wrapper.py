import torch
import time


class MultiModelWrapper(torch.nn.Module):
    def __init__(self, models):
        super(MultiModelWrapper, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x, use_fork=False):
        if use_fork:
            # Parallel execution using torch.jit.fork
            futures = [torch.jit.fork(model, x) for model in self.models]
            return torch.mean(torch.cat([torch.jit.wait(fut) for fut in futures], dim=0), dim=0, keepdim=True)
        else:
            # Sequential execution
            return torch.mean(torch.cat([model(x) for model in self.models], dim=0), dim=0, keepdim=True)


if __name__ == "__main__":
    from model_insightface import load_insightface_ir50_model
    from model_insightface_dropout import load_insightface_ir50_model as load_insightface_ir50_model_dropout

    model_0 = load_insightface_ir50_model(
        r"C:\Users\joaot\Documents\Code\yk-adversarial-attack\models\backbone_ir50_ms1m_epoch120.pth",
        device='cuda:0'
    )
    model_1 = load_insightface_ir50_model_dropout(
        r"C:\Users\joaot\Documents\Code\yk-adversarial-attack\models\backbone_ir50_ms1m_epoch120.pth",
        device='cuda:0'
    )

    wrapper = MultiModelWrapper([model_0, model_1])

    # Benchmark sequential
    start_time = time.time()
    for i in range(20):
        image = torch.randn(1, 3, 112, 112, device='cuda:0')
        b = wrapper(image, use_fork=False)
    print(f"Sequential Time: {time.time() - start_time:.4f} sec")

    del wrapper
    wrapper = MultiModelWrapper([model_0, model_1])

    # Benchmark parallel (torch.jit.fork)
    start_time = time.time()
    for i in range(20):
        image = torch.randn(1, 3, 112, 112, device='cuda:0')
        b = wrapper(image, use_fork=True)
    print(f"Parallel Time (jit.fork): {time.time() - start_time:.4f} sec")
