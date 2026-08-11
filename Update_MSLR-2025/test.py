import torch

if torch.cuda.device_count() >= 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    # DataParallel expects the model to be on the default device or moved afterwards
    # For simplicity, we'll move the base model to the desired device first.
    # DataParallel will handle distributing to other GPUs.
    # model.to(cfg.DEVICE)
    # model = DataParallel(model, device_ids=cfg.GPU_IDS)
# else:
#     model.to(cfg.DEVICE) # Move model to device if not using DataParallel or only one GPU