import torch
import torch.nn as nn
from xresnet1d_101 import xresnet1d101
class KEDEncoderWrapper(nn.Module):
    def __init__(self, ked_encoder, global_dim=512, token_num=100):
        super().__init__()
        self.ked_encoder = ked_encoder                  # 已加载好权重的KED模型
        self.adapt_pool = nn.AdaptiveAvgPool1d(token_num)  # 将token数变为100
        self.proj = nn.Linear(768, global_dim)              # 将768维降为512维

    def forward(self, x):
        feats = self.ked_encoder(x)           # [batch, 768, 157]
        token_emb = self.adapt_pool(feats)    # [batch, 768, 100]
        token_emb = token_emb.transpose(1,2)  # [batch, 100, 768]
        global_emb = token_emb.mean(dim=1)    # [batch, 768]
        global_emb = self.proj(global_emb)    # [batch, 512]
        return global_emb, token_emb

if __name__ == '__main__':
    ked_encoder = xresnet1d101(input_channels=12, kernel_size=5, use_ecgNet_Diagnosis='other')
   
    # 包装成兼容ECG-Chat格式
    ecg_encoder = KEDEncoderWrapper(ked_encoder)
    # forward推理
    x = torch.randn(8, 12, 5000)  # 假设batch=8, 12导联, 5000采样点
    global_emb, token_emb = ecg_encoder(x)
    print(global_emb.shape)  # torch.Size([8, 512])
    print(token_emb.shape)   # torch.Size([8, 100, 768])