import torch.nn as nn
import torch.nn.functional as F
import torch

from ssd.utils import box_utils


# class MultiBoxLoss(nn.Module):
#     def __init__(self, neg_pos_ratio):
#         """Implement SSD MultiBox Loss.

#         Basically, MultiBox loss combines classification loss
#          and Smooth L1 regression loss.
#         """
#         super(MultiBoxLoss, self).__init__()
#         self.neg_pos_ratio = neg_pos_ratio

#     def forward(self, confidence, predicted_locations, labels, gt_locations):
#         """Compute classification loss and smooth l1 loss.

#         Args:
#             confidence (batch_size, num_priors, num_classes): class predictions.
#             predicted_locations (batch_size, num_priors, 4): predicted locations.
#             labels (batch_size, num_priors): real labels of all the priors.
#             gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
#         """
#         num_classes = confidence.size(2)
#         with torch.no_grad():
#             # derived from cross_entropy=sum(log(p))
#             loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
#             mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

#         confidence = confidence[mask, :]
#         classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

#         pos_mask = labels > 0
#         predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
#         gt_locations = gt_locations[pos_mask, :].view(-1, 4)
#         smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
#         num_pos = gt_locations.size(0)
#         return smooth_l1_loss / num_pos, classification_loss / num_pos
    

class ConsistencyLossLR(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(ConsistencyLossLR, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, pred, pred_flip):
        """Compute classification loss and smooth l1 loss.

        Args:
            pred(confidence, predicted_locations, labels, gt_locations): 翻转前预测结果
            pred_flip: 翻转后预测结果
            confidence (batch_size, num_priors, num_classes): class predictions. 第二个维度num_priors表示预选框的数量
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        (confidence, predicted_locations, labels, gt_locations) = pred
        (confidence_flip, predicted_locations_flip, labels_flip, gt_locations_flip) = pred_flip

        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # confidence = confidence[mask, :] # confidence: [1236, 2]
        # confidence_flip = confidence_flip[mask, :]
        # consis_conf = F.kl_div(confidence.view(-1, num_classes).sigmoid(), confidence_flip.view(-1, num_classes).sigmoid(), log_target=True, reduction='mean')
        consis_conf=0

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        predicted_locations_flip = predicted_locations_flip[pos_mask, :].view(-1, 4)

        consis_x = F.mse_loss(predicted_locations[:,0].sigmoid(), (-predicted_locations_flip[:,0]).sigmoid(), reduction='mean')
        consis_y = F.mse_loss(predicted_locations[:,1].sigmoid(), predicted_locations_flip[:,1].sigmoid(), reduction='mean')
        consis_wh = F.mse_loss(predicted_locations[:,2:].sigmoid(), predicted_locations_flip[:,2:].sigmoid(), reduction='mean')
        
        # num_pos = gt_locations.size(0)
        # consis_loss = (consis_conf+consis_x+consis_y+consis_wh)/(4*num_pos)
        consis_loss = (consis_conf+consis_x+consis_y+consis_wh)/3
        
        return consis_loss
    

class ConsistencyLossUD(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(ConsistencyLossUD, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, pred, pred_flip):
        """Compute classification loss and smooth l1 loss.

        Args:
            pred(confidence, predicted_locations, labels, gt_locations): 翻转前预测结果
            pred_flip: 翻转后预测结果
            confidence (batch_size, num_priors, num_classes): class predictions. 第二个维度num_priors表示预选框的数量
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        (confidence, predicted_locations, labels, gt_locations) = pred
        (confidence_flip, predicted_locations_flip, labels_flip, gt_locations_flip) = pred_flip

        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # confidence = confidence[mask, :] # confidence: [1236, 2]
        # confidence_flip = confidence_flip[mask, :]
        # consis_conf = F.kl_div(confidence.view(-1, num_classes).sigmoid(), confidence_flip.view(-1, num_classes).sigmoid(), log_target=True, reduction='mean')
        consis_conf=0

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        predicted_locations_flip = predicted_locations_flip[pos_mask, :].view(-1, 4)

        consis_x = F.mse_loss(predicted_locations[:,0].sigmoid(), predicted_locations_flip[:,0].sigmoid(), reduction='mean')
        consis_y = F.mse_loss(predicted_locations[:,1].sigmoid(), (-predicted_locations_flip[:,1]).sigmoid(), reduction='mean')
        consis_wh = F.mse_loss(predicted_locations[:,2:].sigmoid(), predicted_locations_flip[:,2:].sigmoid(), reduction='mean')
        
        # num_pos = gt_locations.size(0)
        # consis_loss = (consis_conf+consis_x+consis_y+consis_wh)/(4*num_pos)
        consis_loss = (consis_conf+consis_x+consis_y+consis_wh)/3
        
        return consis_loss



class ConsistencyLossLRUD(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(ConsistencyLossLRUD, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, pred, pred_flip):
        """Compute classification loss and smooth l1 loss.

        Args:
            pred(confidence, predicted_locations, labels, gt_locations): 翻转前预测结果
            pred_flip: 翻转后预测结果
            confidence (batch_size, num_priors, num_classes): class predictions. 第二个维度num_priors表示预选框的数量
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        (confidence, predicted_locations, labels, gt_locations) = pred
        (confidence_flip, predicted_locations_flip, labels_flip, gt_locations_flip) = pred_flip

        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # confidence = confidence[mask, :] # confidence: [1236, 2]
        # confidence_flip = confidence_flip[mask, :]
        # consis_conf = F.kl_div(confidence.view(-1, num_classes).sigmoid(), confidence_flip.view(-1, num_classes).sigmoid(), log_target=True, reduction='mean')
        consis_conf=0

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        predicted_locations_flip = predicted_locations_flip[pos_mask, :].view(-1, 4)

        consis_x = F.mse_loss(predicted_locations[:,0].sigmoid(), (-predicted_locations_flip[:,0]).sigmoid(), reduction='mean')
        consis_y = F.mse_loss(predicted_locations[:,1].sigmoid(), (-predicted_locations_flip[:,1]).sigmoid(), reduction='mean')
        consis_wh = F.mse_loss(predicted_locations[:,2:].sigmoid(), predicted_locations_flip[:,2:].sigmoid(), reduction='mean')
        
        # num_pos = gt_locations.size(0)
        # consis_loss = (consis_conf+consis_x+consis_y+consis_wh)/(4*num_pos)
        consis_loss = (consis_conf+consis_x+consis_y+consis_wh)/3
        
        return consis_loss

