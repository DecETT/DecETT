import time
import torch
from loguru import logger as logging

def test(model, test_loader, args):
    logging.info('Start Testing...')
    
    t_start = time.time()
    torch.set_grad_enabled(False)
    model.eval()

    y_true = []
    y_pred_tun = []
    y_pred_proba_tun = []


    print(len(test_loader))

    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(args.device, torch.float), y.to(args.device, torch.long)
        model.to(args.device)
        outputs = model(x)

        if model.__class__.__name__ == "DRL":
            pred_probe_tun = outputs[1]
            pred_tun = outputs[1].argmax(dim=1)
            y_pred_tun.extend(pred_tun.tolist())
            y_pred_proba_tun.extend(pred_probe_tun.tolist())
            y_true.extend(y.tolist())


    if model.__class__.__name__ == "DRL":
        logging.info("Model DRL Testing...")
        val_acc = sum(1 for yt, yp in zip(y_true, y_pred_tun) if yt == yp) / len(y_true)

    t_end = time.time()
    logging.info(f'Test:\nAccuracy: {val_acc:.3f}\n')
    return y_pred_tun, y_pred_proba_tun, t_end - t_start
