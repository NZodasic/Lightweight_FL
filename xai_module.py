import torch
import torch.nn as nn
import torch.autograd as autograd

class SaliencyLearningLoss(nn.Module):
    def __init__(self, lambda_saliency=1.0):
        super(SaliencyLearningLoss, self).__init__()
        self.base_criterion = nn.CrossEntropyLoss()
        self.lambda_saliency = lambda_saliency

    def forward(self, model, x, targets, expert_mask):
        """
        x: Dữ liệu đầu vào, shape (batch, features)
        targets: Nhãn thực tế
        expert_mask: Ma trận nhị phân (Được cung cấp bởi chuyên gia miền hiểu biết/Domain expert)
                     1 = đặc trưng chuyên gia đánh giá là KHÔNG QUAN TRỌNG cần phạt,
                     0 = được phép học.
                     Ma trận này KHÔNG được hệ thống học mà phải truyền vào như tri thức bên ngoài.
                     Ví dụ ROAD dataset: col 0 (ID) = 0, col 1-8 (DATA0-DATA7) = 1.
        """
        # Tránh lỗi in-place mutation, detach x ra khỏi đồ thị và yêu cầu track gradient cho x
        x = x.detach().requires_grad_(True)

        # Forward pass
        logits = model(x)

        # Bug 4 fix: Detach logits cho base_loss để tránh second-order gradient không cần thiết
        # qua CE loss khi loss.backward() được gọi (tiết kiệm memory và tránh instability).
        # Score (để tính gradient saliency) vẫn dùng logits đầy đủ với create_graph=True.
        base_loss = self.base_criterion(logits.detach(), targets)

        # Tính Gradient để dùng làm Saliency Map.
        # Dùng ground truth index kết hợp với gradient của log_softmax thay thế cho raw logits (Theo Ross et al.)
        log_probs = torch.log_softmax(logits, dim=1)
        score = log_probs.gather(1, targets.view(-1, 1)).sum()

        gradients = autograd.grad(outputs=score, inputs=x,
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]

        # Phạt gradient thông qua ma trận nhị phân (expert binary mask matrix)
        # Những gradient rơi vào vùng mask=1 sẽ bị cộng dồn thành loss penalty cực lớn.
        saliency_penalty = torch.sum((gradients * expert_mask) ** 2)

        # Hàm tổng mất mát tích hợp phạt Saliency
        total_loss = base_loss + self.lambda_saliency * saliency_penalty

        return total_loss
