#!/bin/bash
echo "Arthur v2 - Quick Setup"
echo "Using best checkpoint: epoch 2, loss 0.0115"
cp models/arthur_v2_epoch2.pt models/arthur_v2_final.pt
echo "✓ Model ready at models/arthur_v2_final.pt"
echo "✓ Live at: https://arthur-prod.vercel.app"
echo "✓ GitHub: https://github.com/nulljosh/arthur"
