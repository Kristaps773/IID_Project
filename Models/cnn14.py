import torch
import torch.nn as nn
from panns_inference import AudioTagging
import os
import sys

class SpeechCommandNet(nn.Module):
    def __init__(self, n_classes: int = 12):
        super().__init__()
        # Suppress PANNs checkpoint message
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            tagger = AudioTagging(checkpoint_path=None, device='cpu')
            self.backbone = tagger.model
            # Freeze all backbone parameters initially
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            
        # Initialize classifier
        self.classifier = nn.Linear(2048, n_classes)
        
        # Unfreeze conv_block5 and conv_block6
        for name, p in self.backbone.named_parameters():
            if 'conv_block5' in name or 'conv_block6' in name:
                p.requires_grad = True

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # Ensure input is the right shape and type
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Get backbone features
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            out = self.backbone(x)
            
        # Extract embeddings
        if isinstance(out, dict):
            embed = out["embedding"]
        elif isinstance(out, (list, tuple)):
            embed = out[1]
        else:
            raise TypeError(f"Unexpected backbone output: {type(out)}")
            
        # Apply classifier
        return self.classifier(embed) 