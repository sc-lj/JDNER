"""
@Time   :   2021-01-12 15:23:56
@File   :   main.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import argparse
import os
import torch
import pytorch_lightning as pl
from dataset import NerDataset, collate_fn
from torch.utils.data import DataLoader
from models import JDNerTrainingModel
from utils import get_abs_path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='cuda',
                        type=str, help="硬件，cpu or cuda")
    parser.add_argument("--gpu_index", default=0, type=int,
                        help='gpu索引, one of [0,1,2,3,...]')
    parser.add_argument("--load_checkpoint", nargs='?', const=True, default=False, type=str2bool,
                        help="是否加载训练保存的权重, one of [t,f]")
    # parser.add_argument(
    #     '--bert_checkpoint', default='/mnt/disk2/PythonProgram/NLPCode/PretrainModel/chinese_bert_base', type=str)
    parser.add_argument(
        '--bert_checkpoint', default='chinese_bert_base', type=str)
    parser.add_argument('--model_save_path', default='checkpoint', type=str)
    parser.add_argument('--epochs', default=5, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=15, type=int, help='批大小')
    parser.add_argument('--num_workers', default=18,
                        type=int, help='多少进程用于处理数据')
    parser.add_argument('--warmup_epochs', default=8,
                        type=int, help='warmup轮数, 需小于训练轮数')
    parser.add_argument('--lr', default=1e-5, type=float, help='学习率')
    parser.add_argument('--accumulate_grad_batches',
                        default=16,
                        type=int,
                        help='梯度累加的batch数')
    parser.add_argument('--mode', default='train', type=str,
                        help='代码运行模式，以此来控制训练测试或数据预处理，one of [train, test]')
    parser.add_argument('--loss_weight', default=0.8,
                        type=float, help='论文中的lambda，即correction loss的权重')
    parser.add_argument(
        "--label_file", default="data/label2ids.json", help="实体标签id", type=str)
    parser.add_argument(
        "--train_file", default="data/train_data/unlabeled_train_data.txt", help="训练数据集")
    parser.add_argument(
        "--val_file", default="data/pretrain_val_data.txt", help="训练数据集")
    arguments = parser.parse_args()
    # if arguments.hard_device == 'cpu':
    #     arguments.device = torch.device(arguments.hard_device)
    # else:
    #     arguments.device = torch.device(f'cuda:{arguments.gpu_index}')
    if not 0 <= arguments.loss_weight <= 1:
        raise ValueError(
            f"The loss weight must be in [0, 1], but get{arguments.loss_weight}")
    # print(arguments)
    return arguments


def main():
    args = parse_args()
    callbacks = ModelCheckpoint(monitor="step",mode="max",save_top_k=5)
    train_data = NerDataset(args.train_file, args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    # val_data = NerDataset(args.val_file, args)
    # valid_loader = DataLoader(val_data, batch_size=args.batch_size,
    #                           shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    args.number_tag = train_data.number_tag
    args.pylen, args.sklen = train_data.pylen, train_data.sklen
    args.num_labels = len(train_data.get_label_list())
    args.py_num_labels = 430  # 实际语音词汇表里面只有417，外加PAD和UNK，即只有419，但是可以多设几个。
    trainer = pl.Trainer(max_epochs=args.epochs,
                        accelerator = 'dp',
                        plugins=DDPPlugin(find_unused_parameters=True),
                         gpus=[1],
                         accumulate_grad_batches=args.accumulate_grad_batches,
                        # resume_from_checkpoint="lightning_logs/plome_jd/checkpoints/epoch=2-step=12500.ckpt",
                         callbacks=[callbacks,]
                         )
    model = JDNerTrainingModel(args)
    # model.load_from_transformers_state_dict(get_abs_path('checkpoint', 'pytorch_model.bin'))
    # model.load_from_transformers_state_dict(
    #     os.path.join(args.bert_checkpoint, 'pytorch_model.bin'))
    if args.load_checkpoint:
        model.load_state_dict(torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'),
                                         map_location=args.hard_device))
    if args.mode == 'train':
        trainer.fit(model, train_loader)
    # trainer.save_checkpoint("last_model.ckpt")
    # model.load_state_dict(
    #     torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'), map_location=args.hard_device))
    # trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
