import timm
import torch.nn as nn

class EmbeddingGenerator(nn.Module):

    def __init__(self, backbone = 'resnet18', embed_dim = 128):
        super().__init__() # for self.backbone & self.projection

        self.backbone = timm.create_model(model_name=backbone, pretrained=True, num_classes=0) # hold pretrained weights instead of random, removes classification head.

        backbone_dim = self.backbone.num_features 

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, embed_dim)
        )
        self.embed_dim = embed_dim
        print(f"embedding generator: {backbone}: {backbone_dim} -> {embed_dim}")

    def forward(self, x):
        features = self.backbone(x) 
        embeddings = self.projection(features) # shape = (N, 128)
        return embeddings

    def train(self, mode=True):
        super().train(mode)

        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()

        return self

    def freeze_backbone(self): # for transfer learning
        for param in self.backbone.parameters(): 
            param.requires_grad = False
        print("backbone frozen")

    def unfreeze_last_block(self):
        for attr_name in ['layer4', 'blocks']: #layer4 for resnet and blocks for efficientnet
            if hasattr(self.backbone, attr_name):
                block = getattr(self.backbone, attr_name)
                if hasattr(block, '__iter__'):
                    last = list(block)[-1]
                    for param in last.parameters():
                        param.requires_grad = True

                else:
                    for param in block.parameters():
                        param.requires_grad = True
                print("unfroze last block: ", attr_name)
                return
        print("couldn't identify last block. unfreezing the entire backbone")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) # return number of scalar values in that parameter tensor using numel()