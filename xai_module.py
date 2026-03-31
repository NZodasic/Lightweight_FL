import torch
import torch.nn as nn
import torch.autograd as autograd

class DimSeqMaxPoolCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DimSeqMaxPoolCNN, self).__init__()
        # Mạng CNN cơ sở
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, channels, sequence_length)
        """
        features = self.cnn(x)
        
        # Sequence-wise Max-Pooling (Max pool dọc theo chuỗi thời gian/chiều dài)
        # Lấy đặc trưng nổi bật nhất của mỗi channel xuyên suốt thời gian
        seq_pooled, _ = torch.max(features, dim=2) # Shape: (batch_size, channels)
        
        # (Optional) Dimension-wise logic: Đôi khi được áp dụng để lấy đặc trưng chéo kênh
        # dim_pooled, _ = torch.max(seq_pooled, dim=1) 

        # Đưa vào Full Connected để ra dự đoán cuối cùng
        out = self.classifier(seq_pooled)
        return out

class SaliencyLearningLoss(nn.Module):
    def __init__(self, lambda_saliency=1.0):
        super(SaliencyLearningLoss, self).__init__()
        self.base_criterion = nn.CrossEntropyLoss()
        self.lambda_saliency = lambda_saliency

    def forward(self, model, x, targets, expert_mask):
        """
        x: Dữ liệu đầu vào
        targets: Nhãn thực tế
        expert_mask: Ma trận nhị phân (1 = đặc trưng chuyên gia đánh giá là KHÔNG QUAN TRỌNG cần phạt, 
                                      0 = được phép học)
        """
        # Yêu cầu theo dõi gradient cho x để tính Saliency Maps
        x.requires_grad_(True)
        
        # Forward pass
        logits = model(x)
        base_loss = self.base_criterion(logits, targets)
        
        # Tính Gradient của đầu ra đối với đầu vào (đóng vai trò là Saliency Map)
        # Dùng ground truth index để tính gradient
        score = logits.gather(1, targets.view(-1, 1)).sum()
        
        gradients = autograd.grad(outputs=score, inputs=x, 
                                  create_graph=True, retain_graph=True, 
                                  only_inputs=True)[0]
        
        # Phạt gradient thông qua ma trận nhị phân (expert binary mask matrix)
        # Những gradient rơi vào vùng mask=1 sẽ bị cộng dồn thành loss penalty cực lớn.
        saliency_penalty = torch.sum((gradients * expert_mask) ** 2)
        
        # Hàm tổng mất mát tích hợp phạt Saliency
        total_loss = base_loss + self.lambda_saliency * saliency_penalty
        
        return total_loss
