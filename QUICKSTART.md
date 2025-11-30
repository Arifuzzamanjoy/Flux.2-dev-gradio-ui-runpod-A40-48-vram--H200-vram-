# 🚀 Quick Start Guide - FLUX.2-dev Gradio UI

Get started in **5 minutes** with this quick setup guide!

## ⚡ Express Setup (Copy & Paste)

### For RunPod / Cloud GPU

```bash
# 1. Clone repository
git clone https://github.com/Arifuzzamanjoy/Flux.2-dev-gradio-ui-runpod-A40-48-vram--H200-vram-.git
cd Flux.2-dev-gradio-ui-runpod-A40-48-vram--H200-vram-

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers.git

# 5. Set HuggingFace token
export HF_TOKEN="your_token_here"

# 6. Run the application
python "flux2 _gradio_ui.py"
```

### For Local Machine

```bash
# Same as above, but first check:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# If CUDA not available, install NVIDIA drivers and CUDA toolkit first
```

## 📝 Get HuggingFace Token

1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "flux2-ui")
4. Select "Read" permissions
5. Copy the token
6. Accept FLUX.2-dev license: https://huggingface.co/black-forest-labs/FLUX.2-dev

## 🎯 First Generation

1. **Open Browser**: Go to `http://localhost:7860` (or the shared link)
2. **Enter Prompt**: "A professional portrait of a person in studio lighting"
3. **Click Generate**: Wait 10-15 seconds
4. **Download**: Your image is ready!

## 🖼️ Try Multi-Image Input

1. **Upload 2-3 Images**: Click on Image 1, 2, 3 slots
2. **Enter Prompt**: "Combine the style from image 1 with the composition from image 2"
3. **Generate**: FLUX.2 will use all images together!

## 📦 Batch Processing Quick Test

Create a file `test_prompts.json`:
```json
{
  "prompts": [
    "A serene mountain landscape at sunset",
    "A futuristic cityscape with neon lights",
    "An abstract geometric pattern in vibrant colors"
  ]
}
```

1. Go to "Batch Studio" tab
2. Upload `test_prompts.json`
3. Click "Generate Batch"
4. Download ZIP with all 3 images!

## ⚙️ Recommended Settings

### For Speed (Good Quality)
- **Width/Height**: 512x512
- **Steps**: 30
- **Guidance Scale**: 3.5

### For Best Quality
- **Width/Height**: 1024x1024
- **Steps**: 50
- **Guidance Scale**: 4.0

### For Experimentation
- **Width/Height**: 768x768
- **Steps**: 45
- **Guidance Scale**: 3.0-5.0

## 🔧 Common Issues & Solutions

### "Out of Memory" Error
```bash
# Your GPU doesn't have enough VRAM
# Solution: Use smaller dimensions or close other GPU applications
```

### "Model Not Found" Error
```bash
# HuggingFace token not set or invalid
# Solution: 
export HF_TOKEN="your_valid_token"
# OR create .env file:
echo "HF_TOKEN=your_token" > .env
```

### "CUDA Not Available" Error
```bash
# PyTorch not installed with CUDA support
# Solution: Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Slow Generation
```bash
# First generation is always slow (model loading)
# Subsequent generations will be much faster

# Also check:
nvidia-smi  # Verify GPU is being used
```

## 💡 Pro Tips

1. **Use Descriptive Prompts**: More details = better results
   - ❌ "a dog"
   - ✅ "a golden retriever puppy sitting in a garden, professional photography, natural lighting, bokeh background"

2. **Experiment with Guidance Scale**:
   - Lower (2.0-3.5): More creative, artistic
   - Higher (4.0-6.0): Follows prompt more strictly

3. **Multi-Image Workflow**:
   - Image 1: Main style reference
   - Image 2: Composition/pose reference
   - Image 3: Additional context

4. **Save Your Seeds**: Good results? Note the seed for reproducibility!

5. **Batch Processing**: Process 10-50 prompts overnight

## 📈 Performance by GPU

| GPU | VRAM | Recommended Settings |
|-----|------|---------------------|
| RTX 3090 | 24GB | 512x512, steps 30-40 |
| RTX 4090 | 24GB | 768x768, steps 40-50 |
| A40 | 48GB | 1024x1024, steps 50 |
| A100 | 80GB | 1536x1536, steps 50 |
| H100 | 80GB | 2048x2048, steps 50 |

## 🎓 Learning Resources

- **Prompt Engineering**: https://huggingface.co/docs/diffusers/using-diffusers/flux
- **Multi-Image Guide**: Check the "Professional Guide" tab in the UI
- **LoRA Training**: For custom style models

## 🆘 Need Help?

- **GitHub Issues**: Report bugs or request features
- **Documentation**: See full README.md
- **Community**: Share your creations!

---

**🎉 That's it! You're ready to create amazing AI art with FLUX.2-dev!**

Next: Explore the UI tabs for advanced features like batch processing, LoRA models, and session analytics.
