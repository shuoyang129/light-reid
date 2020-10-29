import torch.nn as nn
import sys
import ast
import argparse
import torch

sys.path.append("../..")
import lightreid


# Settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_dir", type=str, default="./results/", help="path to save outputs"
)
parser.add_argument("--dataset", type=str, default="dukemtmcreid", help="")
parser.add_argument(
    "--lightmodel",
    type=ast.literal_eval,
    default=False,
    help="train a small model with model distillation",
)
parser.add_argument(
    "--lightfeat",
    type=ast.literal_eval,
    default=False,
    help="learn binary codes NOT real-value code",
)
parser.add_argument(
    "--lightsearch",
    type=ast.literal_eval,
    default=False,
    help="lightfeat should be True if lightsearch is True",
)

parser.add_argument(
    "--feat_mode",
    type=str,
    default="pooling",
    help="should be in ['pooling','similarity', 'concat']",
)
parser.add_argument(
    "--norm_type", type=str, default="softmax", help="should be in ['softmax', 'l2']"
)
parser.add_argument("--order", type=int, default=1, help="similarity order")
args = parser.parse_args()

assert args.feat_mode in [
    "pooling",
    "similarity",
    "concat",
], "feat_mode must be in ['pooling','similarity', 'concat']"
assert args.norm_type in [
    "l2",
    "softmax",
], "norm_type of BaseReIDModel must be 'l2' or 'softmax'"
# build dataset
DUKE_PATH = "/home/Monday/datasets/DukeMTMC-reID"
datamanager = lightreid.data.DataManager(
    sources=lightreid.data.build_train_dataset([args.dataset]),
    target=lightreid.data.build_test_dataset(args.dataset),
    transforms_train=lightreid.data.build_transforms(
        img_size=[256, 128], transforms_list=["randomflip", "padcrop", "rea"]
    ),
    transforms_test=lightreid.data.build_transforms(
        img_size=[256, 128], transforms_list=[]
    ),
    sampler="pk",
    p=16,
    k=4,
)

# build model
backbone = lightreid.models.backbones.resnet50(pretrained=True, last_stride_one=True)
pooling = nn.AdaptiveAvgPool2d(1)


if args.feat_mode == "concat":
    head = lightreid.models.BNHead(
        in_dim=128 + backbone.dim, class_num=datamanager.class_num
    )
elif args.feat_mode == "pooling":
    head = lightreid.models.BNHead(in_dim=backbone.dim, class_num=datamanager.class_num)
else:
    head = lightreid.models.BNHead(in_dim=128, class_num=datamanager.class_num)

model = lightreid.models.BaseReIDModel(
    backbone=backbone, pooling=pooling, head=head, feat_mode=args.feat_mode,
    norm_type=args.norm_type, order=args.order)

# build loss
criterion = lightreid.losses.Criterion(
    [
        {
            "criterion": lightreid.losses.CrossEntropyLabelSmooth(
                num_classes=datamanager.class_num
            ),
            "weight": 1.0,
        },
        {
            "criterion": lightreid.losses.TripletLoss(
                margin="soft", metric="euclidean"
            ),
            "weight": 1.0,
        },
    ]
)

# build optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00035, weight_decay=5e-4)
lr_scheduler = lightreid.optim.WarmupMultiStepLR(
    optimizer, milestones=[40, 70], gamma=0.1, warmup_factor=0.01, warmup_epochs=10
)
optimizer = lightreid.optim.Optimizer(
    optimizer=optimizer, lr_scheduler=lr_scheduler, max_epochs=120
)

# run
solver = lightreid.engine.Engine(
    results_dir=args.results_dir,
    datamanager=datamanager,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    use_gpu=True,
    light_model=args.lightmodel,
    light_feat=args.lightfeat,
    light_search=args.lightsearch,
)
# train
solver.train(eval_freq=10)
# test
solver.resume_latest_model()
solver.eval(onebyone=True)
# solver.smart_eval(onebyone=True, mode="normal")
# visualize
# solver.visualize()
