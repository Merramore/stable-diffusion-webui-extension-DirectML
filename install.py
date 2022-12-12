#!python3
#install.py

import launch

print("[DirectML] Installing packages.")

if not launch.is_installed("torch_directml"):
    launch.run_pip("install torch-directml==0.1.13.dev221206", "pytorch DirectML")

#TODO: hook cuda device test
#("""
#import torch_directml
#assert torch.is_available()
#""")
