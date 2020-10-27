"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import copy


class BaseReIDModel(nn.Module):
    """
    Archtecture for Based ReID Model
    combine backbone, pooling and head modules
    """

    def __init__(self, backbone, pooling, head, norm_type='softmax'):
        super(BaseReIDModel, self).__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.head = head
        self.use_tanh = False
        self.norm_type = norm_type

    def forward(self, x, y=None, teacher_mode=False, fixcnn=False, test_feat_from_head=True):
        '''
        Args:
            x(torch.tensor): input images
            y(torch.tensor): input labels, used for such as circle loss
        '''
        # cnn backbone
        feats_map = self.backbone(x)
        if fixcnn:
            feats_map = feats_map.detach()
        # pooling
        feats_vec = self.pooling(feats_map).squeeze(3).squeeze(2)
        feats_vec = self.pooling(feats_map).squeeze(3).squeeze(2)
        feats_vec2 = self_similarity2(feats_map, norm_type=self.norm_type)
        feats_vec = torch.cat((feats_vec, feats_vec2), dim=-1)

        # teacher mode
        if teacher_mode:
            headfeats_vec, logits = self.head(
                feats_vec, y, use_tanh=self.use_tanh, teacher_mode=True)
            return feats_vec, headfeats_vec, logits

        # return
        if self.training:
            if self.head.__class__.__name__ == 'BNHead':
                headfeats_vec, logits = self.head(
                    feats_vec, y, use_tanh=self.use_tanh)
                return feats_vec, headfeats_vec, logits
            elif self.head.__class__.__name__ == 'CodePyramid':
                feats_list, headfeats_list, logits_list = self.head(
                    feats_vec, y, use_tanh=self.use_tanh)
                return feats_list, headfeats_list, logits_list
            else:
                assert 0, 'head error'
        else:
            if test_feat_from_head:
                headfeats_vec = self.head(feats_vec, y, use_tanh=self.use_tanh)
                return headfeats_vec
            else:
                return feats_vec

    def enable_tanh(self):
        self.use_tanh = True

    def disable_tanh(self):
        self.use_tanh = False


# class PCBReIDModel(BaseReIDModel):
#
#     def forward(self, x, y=None, fixcnn=False):
#         # conn backbone
#         feat_map = self.backbone(x)
#         if fixcnn:
#             feat_map = feat_map.detach()
#         # pooling
#         feat_vec = self.pooling(feat_map).squeeze(3)
#         # return
#         if self.training:
#             embedding_list, logits_list = self.head(feat_vec, y)
#             return embedding_list, logits_list
#         else:
#             feat = self.head(feat_vec, y)
#             return feat
def self_similarity(featuremaps):
    b, d, h, w = featuremaps.shape
    # print("featuremaps shape:", featuremaps.shape, "==>", (b, d, h, w))
    kernels = featuremaps.permute([0, 2, 3, 1])  # [b, h, w, d]
    similarities = torch.zeros(b, 2 * h*w, device=featuremaps.device)
    for i in range(b):
        kernel = kernels[i]  # [h, w, d]
        featuremap = featuremaps[i].unsqueeze_(0)  # [1, d, h, w]
        # print("featuremap shape:", featuremap.shape)
        kernel = kernel.reshape(-1, d, 1, 1)  # [hw, d, 1, 1]
        similarity = F.conv2d(featuremap, kernel)  # [1,hw, h, w]
        # print("similarity shape: ", similarity.shape)
        similarity = similarity.view(1, h*w, h*w)
        # print("similarity shape: ", similarity.shape)
        similarity = torch.cat(
            (similarity.max(dim=1)[0], similarity.max(dim=2)[0]), dim=-1)
        # print("similarity shape: ", similarity.shape)
        similarities[i] = similarity
    return similarities


def self_similarity2(featuremaps, norm_type='softmax'):
    # norm_type: l1 or softmax
    assert norm_type in [
        'l1', 'softmax'], "norm_type of BaseReIDModel should be 'l1' or 'softmax'"
    b, d, h, w = featuremaps.shape
    # print("featuremaps shape:", featuremaps.shape, "==>", (b, d, h, w))
    kernels = featuremaps.permute([0, 2, 3, 1])  # [b, h, w, d]
    similarities = torch.zeros(b, h*w, device=featuremaps.device)
    # similarities = torch.zeros(b, 2 * h*w, device=featuremaps.device)
    for i in range(b):
        kernel = kernels[i]  # [h, w, d]
        featuremap = featuremaps[i].reshape(-1, d)  # [d, h, w]
        # print("featuremap shape:", featuremap.shape)
        kernel = kernel.reshape(d, -1)  # [hw, d, 1, 1]
        similarity = torch.matmul(featuremap, kernel)  # [hw, hw]
        # print("similarity shape: ", similarity.shape)
        if norm_type == 'l1':
            similarity = similarity/torch.norm(similarity, p=1, dim=-1).expand(
                similarity.size()[1], similarity.size()[0]).transpose(0, 1)
        elif norm_type == 'softmax':
            similarity = F.softmax(similarity, dim=-1)
        # similarity = similarity/torch.norm(similarity)
        # similarity = F.softmax(similarity)
        # print(similarity.sum(-1))
        similarity = similarity.view(1, h*w, h*w)
        # print("similarity shape: ", similarity.shape)
        # similarity = torch.cat(
        #     (similarity.max(dim=1)[0], similarity.max(dim=2)[0]), dim=-1)
        similarity = similarity.max(dim=-1)[0]
        # print("similarity shape: ", similarity.shape)
        similarities[i] = similarity
    return similarities


if __name__ == '__main__':
    import numpy as np
    tem = np.random.random(3*5*2*5)
    featuremaps = torch.tensor(tem).reshape((3, 5, 2, 5)).cuda()
    similarities = self_similarity2(featuremaps.float())
    # similarities = self_similarity2(featuremaps.float(), 'softmax')
    print(similarities)
