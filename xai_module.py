import torch
import torch.nn as nn
import torch.autograd as autograd

class DimSeqMaxPoolCNN(nn.Module):
    def __init__(self, input_channels, num_classes, seq_length=9):
        super(DimSeqMaxPoolCNN, self).__init__()
        # Mạng CNN cơ sở (Parallel CNNs theo Ghaeini et al.)
        self.cnn_3 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cnn_5 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Trọng số tuyến tính sẽ nhận đầu vào được biểu diễn từ cả hai nhánh, 
        # bao gồm cả sequence-wise (64) và dimension-wise (seq_length) cho từng nhánh CNN
        self.classifier = nn.Linear(128 + 2 * seq_length, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, channels, sequence_length)
        """
        features_3 = self.cnn_3(x)
        features_5 = self.cnn_5(x)
        
        # Sequence-wise Max-Pooling (Max pool dọc theo chuỗi thời gian/chiều dài, dim=2) -> Shape: (batch_size, channels)
        seq_pooled_3, _ = torch.max(features_3, dim=2)
        seq_pooled_5, _ = torch.max(features_5, dim=2)
        
        # Dimension-wise Max-Pooling (Max pool dọc theo không gian kênh, dim=1) -> Shape: (batch_size, sequence_length)
        dim_pooled_3, _ = torch.max(features_3, dim=1)
        dim_pooled_5, _ = torch.max(features_5, dim=1)

        # Ghép tất cả các biểu diễn lại với nhau
        concat_features = torch.cat([seq_pooled_3, dim_pooled_3, seq_pooled_5, dim_pooled_5], dim=1)

        # Đưa vào Full Connected để ra dự đoán cuối cùng
        out = self.classifier(concat_features)
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
        expert_mask: Ma trận nhị phân (Được cung cấp bởi chuyên gia miền hiểu biết/Domain expert)
                     1 = đặc trưng chuyên gia đánh giá là KHÔNG QUAN TRỌNG cần phạt, 
                     0 = được phép học.
                     Ma trận này KHÔNG được hệ thống học mà phải truyền vào như tri thức bên ngoài.
        """
        # Tránh lỗi in-place mutation, detach x ra khỏi đồ thị và yệu cầu track gradient cho x
        x = x.detach().requires_grad_(True)
        
        # Forward pass
        logits = model(x)
        base_loss = self.base_criterion(logits, targets)
        
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
