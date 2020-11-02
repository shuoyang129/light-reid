"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import numpy as np
import os
import time
import torch
import torch.nn as nn
from prettytable import PrettyTable

from lightreid.evaluations import (
    PreRecEvaluator,
    CmcMapEvaluator,
    CmcMapEvaluator1b1,
    CmcMapEvaluatorC2F,
    accuracy,
)
from lightreid.utils import (
    MultiItemAverageMeter,
    CatMeter,
    AverageMeter,
    Logging,
    time_now,
    os_walk,
)
from lightreid.visualizations import visualize_ranked_results
import lightreid


class Engine(object):
    """
    Engine for light-reid training
    Args：
        necessary:
            results_dir(str): path to save results
            datamanager(lighreid.data.DataManager): provide train_loader and test_loader
            model(lightreid.model.BaseReIDModel):
            criterion(ligherreid.loss.Criterion): compute losses
            optimizer(lightreid.optim.Optimizer):
            use_gpu(bool): use CUDA if True
        optional:
            light_model(bool): if True, use distillation to learn a small model.
            light_feat(bool): if True, learn binary codes and evaluate with hamming distance.
            light_search(bool): if True, use pyramid head lean multiple codes, and search with coarse2fine.
    """

    def __init__(
        self,
        results_dir,
        datamanager,
        model,
        criterion,
        optimizer,
        use_gpu,
        data_parallel=False,
        sync_bn=False,
        eval_metric="cosine",
        light_model=False,
        light_feat=False,
        light_search=False,
    ):

        # base settings
        self.results_dir = os.path.join(
            results_dir,
            "lightmodel({})-lightfeat({})-lightsearch({})".format(
                light_model, light_feat, light_search
            ),
        )
        self.datamanager = datamanager
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.eval_metric = eval_metric

        self.loss_meter = MultiItemAverageMeter()
        os.makedirs(self.results_dir, exist_ok=True)
        self.logging = Logging(os.path.join(self.results_dir, "logging.txt"))

        # optinal settings for light-reid learning
        self.light_model = light_model
        self.light_feat = light_feat
        self.light_search = light_search
        self.logging("\n" + "****" * 5 + " light-reid settings " + "****" * 5)
        self.logging("light_model:  {}".format(light_model))
        self.logging("light_feat:  {}".format(light_feat))
        self.logging("light_search:  {}".format(light_search))
        self.logging("****" * 5 + " light-reid settings " + "****" * 5 + "\n")

        # if enable light_model, learn small model with distillation
        # update model to a small model (res18)
        # load teacher model from model_teacher (should be trained before)
        # add KLLoss(distillation loss) to criterion
        if self.light_model:
            # load teacher model
            teacher_path = os.path.join(
                results_dir,
                "lightmodel(False)-lightfeat(False)-lightsearch(False)/final_model.pth.tar",
            )
            assert os.path.exists(
                teacher_path
            ), "lightmodel was enabled, expect {} as a teachder but file not exists".format(
                teacher_path
            )
            model_t = torch.load(teacher_path, map_location=torch.device("cpu"))
            self.model_t = model_t.to(
                self.device
            ).eval()  # set teacher as evaluation mode
            self.logging(
                "[light_model was enabled] load teacher model from {}".format(
                    teacher_path
                )
            )
            # modify model to a small model (ResNet18 as default here)
            pretrained, last_stride_one = (
                self.model.backbone.pretrained,
                self.model.backbone.last_stride_one,
            )
            self.model.backbone.__init__("resnet18", pretrained, last_stride_one)
            self.model.head.__init__(
                self.model.backbone.dim,
                self.model.head.class_num,
                self.model.head.classifier,
            )
            if self.model.head.classifier.__class__.__name__ == "Circle":
                self.model.head.classifier.__init__(
                    self.model.backbone.dim,
                    self.model.head.classifier._num_classes,
                    self.model.head.classifier._s,
                    self.model.head.classifier._m,
                )
            elif self.model.head.classifier.__class__.__name__ == "Linear":
                self.model.head.classifier.__init__(
                    self.model.backbone.dim,
                    self.model.head.classifier.out_features,
                    self.model.head.classifier.bias,
                )
            else:
                assert 0, "classifier error"
            self.logging("[light_model was enabled] modify model to ResNet18")
            # update optimizer
            optimizer_defaults = self.optimizer.optimizer.defaults
            self.optimizer.optimizer.__init__(
                self.model.parameters(), **optimizer_defaults
            )
            self.logging("[light_model was enabled] update optimizer parameters")
            # add KLLoss to criterion
            self.criterion.criterion_list.append(
                {"criterion": lightreid.losses.KLLoss(t=4), "weight": 1.0}
            )
            self.logging("[light_model was enabled] add KLLoss for model distillation")

        # if enable light_feat,
        # learn binary codes NOT real-value features
        # evaluate with hamming metric, NOT cosine NEITHER euclidean metrics
        if self.light_feat:
            self.model.enable_tanh()
            self.eval_metric = "hamming"
            self.logging(
                "[light_feat was enabled] model learn binary codes, and is evluated with hamming distance"
            )
            self.logging(
                "[light_feat was enabled] update eval_metric from {} to hamming by setting self.eval_metric=hamming".format(
                    eval_metric
                )
            )

        # if enable light_search,
        # learn binary codes of multiple length with pyramid-head
        # and search with coarse2fine strategy
        if self.light_search:
            # modify head to pyramid-head
            in_dim, class_num = self.model.head.in_dim, self.model.head.class_num
            head_type, classifier_type = (
                self.model.head.__class__.__name__,
                self.model.head.classifier.__class__.__name__,
            )
            self.model.head = lightreid.models.CodePyramid(
                in_dim=in_dim,
                out_dims=[2048, 512, 128, 32],
                class_num=class_num,
                head=head_type,
                classifier=classifier_type,
            )
            self.logging(
                "[light_search was enabled] learn multiple codes with {}".format(
                    self.model.head.__class__.__name__
                )
            )
            # update optimizer parameters
            optimizer_defaults = self.optimizer.optimizer.defaults
            self.optimizer.optimizer.__init__(
                self.model.parameters(), **optimizer_defaults
            )
            self.logging("[light_search was enabled] update optimizer parameters")
            # add self-ditillation loss loss for pyramid-haed
            self.criterion.criterion_list.extend(
                [
                    {
                        "criterion": lightreid.losses.ProbSelfDistillLoss(),
                        "weight": 1.0,
                    },
                    {
                        "criterion": lightreid.losses.SIMSelfDistillLoss(),
                        "weight": 1000.0,
                    },
                ]
            )
            self.logging(
                "[light_search was enabled] add ProbSelfDistillLoss and SIMSelfDistillLoss"
            )

        self.model = self.model.to(self.device)
        if data_parallel:
            if not sync_bn:
                self.model = nn.DataParallel(self.model)
            if sync_bn:
                # torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                # self.model = nn.parallel.DistributedDataParallel(self.model)

    def save_model(self, save_epoch):
        """
        save model parameters (only state_dict) in self.results_dir/model_{epoch}.pth
        save model (architecture and state_dict) in self.results_dir/final_model.pth.tar, may be used as a teacher
        """
        model_path = os.path.join(self.results_dir, "model_{}.pth".format(save_epoch))
        torch.save(self.model.state_dict(), model_path)
        root, _, files = os_walk(self.results_dir)
        pth_files = [
            file for file in files if ".pth" in file and file != "final_model.pth.tar"
        ]
        if len(pth_files) > 1:
            pth_epochs = sorted(
                [
                    int(pth_file.replace(".pth", "").split("_")[1])
                    for pth_file in pth_files
                ],
                reverse=False,
            )
            model_path = os.path.join(root, "model_{}.pth".format(pth_epochs[0]))
            os.remove(model_path)
        torch.save(self.model, os.path.join(self.results_dir, "final_model.pth.tar"))

    def resume_model(self, model_path, strict=True):
        """
        resume from model_path
        """
        self.model.load_state_dict(torch.load(model_path), strict=strict)

    def resume_latest_model(self):
        """
        resume from the latest model in path self.results_dir
        """
        root, _, files = os_walk(self.results_dir)
        pth_files = [
            file for file in files if ".pth" in file and file != "final_model.pth.tar"
        ]
        if len(pth_files) != 0:
            pth_epochs = [
                int(pth_file.replace(".pth", "").split("_")[1])
                for pth_file in pth_files
            ]
            max_epoch = max(pth_epochs)
            model_path = os.path.join(root, "model_{}.pth".format(max_epoch))
            self.model.load_state_dict(torch.load(model_path), strict=True)
            self.logging(time_now(), "restore from {}".format(model_path))
            return max_epoch
        else:
            return None

    def set_train(self):
        """
        set model as training mode
        """
        self.model = self.model.train()

    def set_eval(self):
        """
        set mode as evaluation model
        """
        self.model = self.model.eval()

    def train(self, auto_resume=True, eval_freq=0):
        """
        Args:
            auto_resume(boolean): automatically resume latest model from self.result_dir/model_{latest_epoch}.pth if True.
            eval_freq(int): if type is int, evaluate every eval_freq. default is 0.
        """

        # automatically resume from the latest model
        start_epoch = 0
        if auto_resume:
            start_epoch = self.resume_latest_model()
            start_epoch = 0 if start_epoch is None else start_epoch
        # train loop
        for curr_epoch in range(start_epoch, self.optimizer.max_epochs):
            # save model
            self.save_model(curr_epoch)
            # evaluate final model
            if eval_freq > 0 and curr_epoch % eval_freq == 0 and curr_epoch > 0:
                self.eval(onebyone=False)
            # train
            results = self.train_an_epoch(curr_epoch)
            # logging
            self.logging(EPOCH=curr_epoch, TIME=time_now(), RESULTS=results)
        # save final model
        self.save_model(self.optimizer.max_epochs)
        # evaluate final model
        self.eval(onebyone=False)

    def train_an_epoch(self, epoch):

        self.set_train()
        self.loss_meter.reset()

        for idx, batch in enumerate(self.datamanager.train_loader):
            # load batch data
            imgs, pids, camids = batch
            imgs, pids, camids = (
                imgs.to(self.device),
                pids.to(self.device),
                camids.to(self.device),
            )
            # forward
            fix_cnn = (
                epoch < self.optimizer.fix_cnn_epochs
                if hasattr(self, "fix_cnn_epochs")
                else False
            )
            feats, bnfeats, logits = self.model(imgs, pids, fixcnn=fix_cnn)
            acc = accuracy(logits[0], pids, [1])[0]
            # teacher model
            if self.light_model:
                with torch.no_grad():
                    feats_t, headfeats_t, logits_t = self.model_t(
                        imgs, pids, teacher_mode=True
                    )
                loss, loss_dict = self.criterion.compute(
                    feats=feats,
                    head_feats=bnfeats,
                    logits=logits[0],
                    pids=pids,
                    logits_s=logits[1],
                    logits_t=logits_t[1],
                )
            else:
                loss, loss_dict = self.criterion.compute(
                    feats=feats, head_feats=bnfeats, logits=logits[0], pids=pids
                )
            loss_dict["Accuracy"] = acc
            # optimize
            self.optimizer.optimizer.zero_grad()
            loss.backward()
            self.optimizer.optimizer.step()
            # update learning rate
            self.optimizer.lr_scheduler.step(epoch)
            # record
            self.loss_meter.update(loss_dict)

        # learning rate
        self.loss_meter.update({"LR": self.optimizer.optimizer.param_groups[0]["lr"]})

        return self.loss_meter.get_str()

    def eval(self, onebyone=False, return_pr=False, return_vislist=False):
        """
        Args:
            onebyone(bool): evaluate query one by one, otherwise in a parallel way
        """

        metric = self.eval_metric
        self.set_eval()

        table = PrettyTable(["dataset", "map", "rank-1", "rank-5", "rank-10"])
        for (
            dataset_name,
            (query_loader, gallery_loader),
        ) in self.datamanager.query_gallery_loader_dict.items():

            # extract features
            time_meter = AverageMeter()
            query_feats, query_pids, query_camids = self.extract_feats(
                query_loader, time_meter=time_meter
            )
            gallery_feats, gallery_pids, gallery_camids = self.extract_feats(
                gallery_loader, time_meter=time_meter
            )
            self.logging(
                "[Feature Extraction] feature extraction time per batch (64) is {}s".format(
                    time_meter.get_val()
                )
            )

            # compute mAP and rank@k
            if isinstance(query_feats, np.ndarray):  #
                if not onebyone:  # eval all query images one shot
                    mAP, CMC = CmcMapEvaluator(
                        metric=metric, mode="inter-camera"
                    ).evaluate(
                        query_feats,
                        query_camids,
                        query_pids,
                        gallery_feats,
                        gallery_camids,
                        gallery_pids,
                    )
                else:  # eval query images one by one
                    mAP, CMC, ranktime, evaltime = CmcMapEvaluator1b1(
                        metric=metric, mode="inter-camera"
                    ).compute(
                        query_feats,
                        query_camids,
                        query_pids,
                        gallery_feats,
                        gallery_camids,
                        gallery_pids,
                        # return_time=True,
                    )
            elif isinstance(query_feats, list):  # eval with coarse2fine
                mAP, CMC, ranktime, evaltime = CmcMapEvaluatorC2F(
                    metric=metric, mode="inter-camera"
                ).compute(
                    query_feats,
                    query_camids,
                    query_pids,
                    gallery_feats,
                    gallery_camids,
                    gallery_pids,
                    # return_time=True,
                )

            # compute precision-recall curve
            if return_pr:
                pr_evaluator = PreRecEvaluator(metric=metric, mode="inter-camera")
                pres, recalls, thresholds = pr_evaluator.evaluate(
                    query_feats,
                    query_camids,
                    query_pids,
                    gallery_feats,
                    gallery_camids,
                    gallery_pids,
                )
                pr_evaluator.plot_prerecall_curve(self.results_dir, pres, recalls)

            # logging
            table.add_row(
                [dataset_name, str(mAP), str(CMC[0]), str(CMC[4]), str(CMC[9])]
            )
            self.logging(mAP, CMC[:150])
            try:
                self.logging(
                    "average ranking time: {}ms; average evaluation time: {}ms".format(
                        ranktime * 1000, evaltime * 1000
                    )
                )
            except:
                pass
        self.logging(table)

    def smart_eval(
        self, onebyone=False, return_pr=False, return_vislist=False, mode="normal"
    ):
        """
        Args:
            onebyone(bool): evaluate query one by one, otherwise in a parallel way
        """

        assert mode in [
            "normal",
            "extract_only",
            "eval_only",
        ], "expect mode in normal, extract_only and eval_only, but got {}".format(mode)
        metric = self.eval_metric
        self.set_eval()

        # extract features
        if mode == "extract_only" or mode == "normal":
            time_meter = AverageMeter()
            query_feats, query_pids, query_camids = self.extract_feats(
                self.datamanager.query_loader, time_meter=time_meter
            )
            gallery_feats, gallery_pids, gallery_camids = self.extract_feats(
                self.datamanager.gallery_loader, time_meter=time_meter
            )
            self.logging(
                "[Feature Extraction] feature extraction time per batch (64) is {}s".format(
                    time_meter.get_val()
                )
            )
            features = {
                "query_feats": query_feats,
                "query_pids": query_pids,
                "query_camids": query_camids,
                "gallery_feats": gallery_feats,
                "gallery_pids": gallery_pids,
                "gallery_camids": gallery_camids,
            }
            torch.save(features, os.path.join(self.results_dir, "features.pkl"))
            return
        elif mode == "eval_only" or mode == "normal":
            assert os.path.exists(
                os.path.join(self.results_dir, "features.pkl")
            ), "expect load features from file {} but file not exists".format(
                os.path.join(self.results_dir, "features.pkl")
            )
            features = torch.load(os.path.join(self.results_dir, "features.pkl"))
            query_feats = features["query_feats"]
            query_pids = features["query_pids"]
            query_camids = features["query_camids"]
            gallery_feats = features["gallery_feats"]
            gallery_pids = features["gallery_pids"]
            gallery_camids = features["gallery_camids"]

            # compute mAP and rank@k
            if isinstance(query_feats, np.ndarray):  #
                if not onebyone:  # eval all query images one shot
                    mAP, CMC = CmcMapEvaluator(
                        metric=metric, mode="inter-camera"
                    ).evaluate(
                        query_feats,
                        query_camids,
                        query_pids,
                        gallery_feats,
                        gallery_camids,
                        gallery_pids,
                    )
                else:  # eval query images one by one
                    mAP, CMC, ranktime, evaltime = CmcMapEvaluator1b1(
                        metric=metric, mode="inter-camera"
                    ).compute(
                        query_feats,
                        query_camids,
                        query_pids,
                        gallery_feats,
                        gallery_camids,
                        gallery_pids,
                        return_time=True,
                    )
            elif isinstance(query_feats, list):  # eval with coarse2fine
                mAP, CMC, ranktime, evaltime = CmcMapEvaluatorC2F(
                    metric=metric, mode="inter-camera"
                ).compute(
                    query_feats,
                    query_camids,
                    query_pids,
                    gallery_feats,
                    gallery_camids,
                    gallery_pids,
                    return_time=True,
                )

            # compute precision-recall curve
            if return_pr:
                pr_evaluator = PreRecEvaluator(metric=metric, mode="inter-camera")
                pres, recalls, thresholds = pr_evaluator.evaluate(
                    query_feats,
                    query_camids,
                    query_pids,
                    gallery_feats,
                    gallery_camids,
                    gallery_pids,
                )
                pr_evaluator.plot_prerecall_curve(self.results_dir, pres, recalls)

            self.logging(mAP, CMC[:150])
            try:
                self.logging(
                    "average ranking time: {} ms; average evaluation time: {} ms".format(
                        ranktime * 1000, evaltime * 1000
                    )
                )
            except:
                pass
            return mAP, CMC[:150]

    def visualize(self):
        import sklearn.metrics.pairwise as skp

        metric = self.eval_metric
        self.set_eval()
        query_feats, query_pids, query_camids = self.extract_feats(
            self.datamanager.query_loader
        )
        gallery_feats, gallery_pids, gallery_camids = self.extract_feats(
            self.datamanager.gallery_loader
        )
        if metric == "cosine":
            distmat = skp.cosine_distances(
                query_feats, gallery_feats
            )  # please note, it is cosine distance not similarity
        elif metric == "euclidean":
            distmat = skp.euclidean_distances(query_feats, gallery_feats)
        elif metric == "hamming":
            distmat = lightreid.utils.hamming_distance(query_feats, gallery_feats)
        dataset = [
            self.datamanager.query_dataset.samples,
            self.datamanager.gallery_dataset.samples,
        ]
        visualize_ranked_results(
            distmat,
            dataset,
            save_dir="./vis-results/",
            topk=20,
            mode="inter-camera",
            show="all",
        )

    def extract_feats(self, loader, feat_from_head=True, time_meter=None):

        self.set_eval()

        # compute features
        features_meter = None
        pids_meter, camids_meter = CatMeter(), CatMeter()
        with torch.no_grad():
            for batch in loader:
                imgs, pids, cids = batch
                imgs, pids, cids = (
                    imgs.to(self.device),
                    pids.to(self.device),
                    cids.to(self.device),
                )
                if time_meter is not None:
                    torch.cuda.synchronize()
                    ts = time.time()
                feats = self.model(imgs, test_feat_from_head=feat_from_head)
                if time_meter is not None:
                    torch.cuda.synchronize()
                    time_meter.update(time.time() - ts)
                if isinstance(feats, torch.Tensor):
                    if features_meter is None:
                        features_meter = CatMeter()
                    features_meter.update(feats.data)
                elif isinstance(feats, list):
                    if features_meter is None:
                        features_meter = [CatMeter() for _ in range(len(feats))]
                    for idx, feats_i in enumerate(feats):
                        features_meter[idx].update(feats_i.data)
                else:
                    assert 0
                pids_meter.update(pids.data)
                camids_meter.update(cids.data)

        if isinstance(features_meter, list):
            feats = [val.get_val_numpy() for val in features_meter]
        else:
            feats = features_meter.get_val_numpy()
        pids = pids_meter.get_val_numpy()
        camids = camids_meter.get_val_numpy()

        return feats, pids, camids

