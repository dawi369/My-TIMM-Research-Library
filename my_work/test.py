import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import timm



model = timm.create_model('resnet50d', pretrained=True)

