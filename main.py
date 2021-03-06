from core.utils import parse_args, setup_determinism
from core.utils import build_loss_func, build_optim
from core.utils import build_scheduler, load_checkpoint
from core.config import get_cfg_defaults
from core.dataset import build_dataloader
from core.model import build_model, train_loop, valid_model, test_model
# from core.model import valid_model_macro

# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import torch.nn as nn

import os

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


# SET UP GLOBAL VARIABLE
scaler = GradScaler()


def main(cfg, args):
    # Setup logger
    # sum_writer = SummaryWriter(f"test2")

    # Declare variables
    best_metric = 0
    start_epoch = 0
    mode = args.mode

    # Setup folder
    if not os.path.isdir(cfg.DIRS.WEIGHTS):
        os.mkdir(cfg.DIRS.WEIGHTS)

    # Load Data
    trainloader = build_dataloader(cfg, mode="train")
    validloader = build_dataloader(cfg, mode="valid")
    # testloader = build_dataloader(cfg, mode="test")

    # Define model/loss/optimizer/Scheduler
    model = build_model(cfg)
    # model = nn.DataParallel(model)
    # loss = build_loss_func(cfg)
    weight=torch.Tensor([0.1, 0.1, 0.1, 1.5])
    weight = weight.to("cuda")
    loss = nn.CrossEntropyLoss(weight=weight)
    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(args, len(trainloader), cfg)
    # Load model checkpoint
    model, start_epoch, best_metric = load_checkpoint(args, model)
    start_epoch = 0
    best_metric = 0
    if cfg.SYSTEM.GPU:
        model = model.cuda()

    # Run Script
    if mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHES):
            print("EPOCH", epoch)
            train_loss = train_loop(
                cfg,
                epoch,
                model,
                trainloader,
                loss,
                scheduler,
                optimizer,
                scaler
            )
            best_metric = valid_model(
                cfg,
                mode,
                epoch,
                model,
                validloader,
                loss,
                best_metric=best_metric,
                save_prediction= False,
                visual=False
            )

    elif mode == "valid":
        valid_model(
            cfg, mode, 0, model, validloader, loss, best_metric=best_metric, save_prediction=True, visual=True
        )
    elif mode == "test":
        test_model(cfg, mode, model, validloader, loss)


if __name__ == "__main__":
    # Set up Variable
    seed = 10

    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)

    # Set seed for reproducible result
    setup_determinism(seed)

    main(cfg, args)
