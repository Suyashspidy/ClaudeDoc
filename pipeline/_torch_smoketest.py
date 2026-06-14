import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list())
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    import time
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    print(f"10x 4096x4096 matmul on GPU OK: {(time.time()-t)*1000:.0f} ms total")
    print("result sum:", float(c.sum()))
