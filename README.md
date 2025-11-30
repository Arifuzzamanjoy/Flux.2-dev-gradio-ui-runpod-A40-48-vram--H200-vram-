# FLUX.2-dev Gradio UI

🎨 **Professional AI Image Generation Studio** powered by FLUX.2-dev with multi-image input support, 4-bit quantization, and batch processing.

![Version](https://img.shields.io/badge/version-2.1.0_Pro-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![GPU](https://img.shields.io/badge/GPU-CUDA_Required-red)

## ✨ Features

### 🚀 Core Capabilities
- **FLUX.2-dev Integration** - Latest 32B parameter model with built-in text encoder
- **4-bit Quantization** - Memory-efficient inference (~18GB VRAM vs 64GB full model)
- **Multi-Image Input** - Upload up to 3 reference images for context-aware generation
- **Batch Processing** - Generate multiple images from JSON/text prompt files
- **LoRA Support** - Style enhancement with custom LoRA models
- **Professional UI** - Clean Gradio interface with advanced controls

### 🎯 Multi-Image Magic
- **Style Transfer** - Combine styles from different images
- **Character Consistency** - Reference faces, poses, and backgrounds separately
- **Image Editing** - Context-aware modifications using reference images
- **No Fine-tuning Required** - Direct character/object/style reference

### ⚡ Optimizations
- **Automatic Fallback** - FLUX.2-dev 4-bit → Full FLUX.2-dev → FLUX.1-dev
- **Memory Management** - Smart caching and GPU optimization
- **Session Analytics** - Track generation stats and performance
- **Error Handling** - Robust validation and fallback mechanisms

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support
  - **Minimum**: 24GB VRAM (for 4-bit quantized model)
  - **Recommended**: 40GB+ VRAM (A40, A100, H100, H200)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space for model weights

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Arifuzzamanjoy/Flux.2-dev-gradio-ui-runpod-A40-48-vram--H200-vram-.git
cd Flux.2-dev-gradio-ui-runpod-A40-48-vram--H200-vram-
```

### Step 2: Create Virtual Environment
```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install diffusers from source (for latest FLUX.2 support)
pip install git+https://github.com/huggingface/diffusers.git
```

### Step 4: Configure HuggingFace Token
```bash
# Create .env file
echo "HF_TOKEN=your_huggingface_token_here" > .env

# OR set environment variable
export HF_TOKEN="your_huggingface_token_here"
```

**Get your HuggingFace token:**
1. Visit https://huggingface.co/settings/tokens
2. Create a new token with read permissions
3. Accept FLUX.2-dev license at https://huggingface.co/black-forest-labs/FLUX.2-dev

### Step 5: Run the Application
```bash
python "flux2 _gradio_ui.py"
```

The UI will open in your browser at `http://localhost:7860`

## 🎮 Usage Guide

### Single Image Generation
1. **Enter Prompt**: Describe your desired image in detail
2. **Upload References** (Optional): Add up to 3 reference images
3. **Adjust Settings**: Modify width, height, steps, guidance scale
4. **Select LoRA** (Optional): Choose style enhancement model
5. **Generate**: Click "Generate Professional Images"

### Multi-Image Example Prompts
```
"Combine the art style from image 1 with the composition from image 2"

"Show the character from image 1 in the pose from image 2 with background from image 3"

"Apply the lighting from image 1 to the scene in image 2"
```

### Batch Processing
1. **Prepare Prompt File**: Create JSON or text file with prompts
   ```json
   {
     "prompts": [
       "A serene landscape at sunset",
       "A futuristic cityscape at night",
       "An abstract digital art piece"
     ]
   }
   ```
2. **Upload File**: Use batch processing tab
3. **Configure Settings**: Set batch parameters
4. **Generate**: Process all prompts automatically
5. **Download**: Get ZIP file with all generated images

## ⚙️ Configuration

### Model Selection
The script automatically tries to load models in this order:
1. **FLUX.2-dev-bnb-4bit** (Default) - 4-bit quantized, ~18GB VRAM
2. **FLUX.2-dev** (Fallback) - Full model, ~64GB VRAM
3. **FLUX.1-dev** (Final Fallback) - Older version

### Memory Optimization
```python
# In the script, adjust cache directory if needed:
CACHE_DIR = os.path.expanduser("/root/.cache")  # Change if needed
```

### Generation Parameters
- **Width/Height**: 256-3048 (must be divisible by 64)
- **Steps**: 10-150 (45 recommended for quality)
- **Guidance Scale**: 1.0-8.0 (3.5 recommended)
- **Seed**: -1 for random, or specific number for reproducibility

## 📊 Performance Benchmarks

| GPU Model | VRAM | Model Version | Generation Time (512x512) |
|-----------|------|---------------|---------------------------|
| RTX 4090  | 24GB | 4-bit         | ~8-12 seconds            |
| A40       | 48GB | 4-bit         | ~6-10 seconds            |
| A100      | 80GB | Full          | ~5-8 seconds             |
| H100      | 80GB | Full          | ~3-5 seconds             |

## 🔧 Troubleshooting

### Out of Memory Error
```bash
# Use 4-bit quantized model (automatic)
# Or reduce image dimensions
# Or decrease batch size
```

### Model Download Issues
```bash
# Check HuggingFace token
huggingface-cli whoami

# Manually download models
huggingface-cli download diffusers/FLUX.2-dev-bnb-4bit
```

### CUDA Not Available
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Gradio Connection Issues
```bash
# Run with share link
# The script automatically uses share=True

# Or specify port
# Modify last line in script: demo.launch(server_port=7860)
```

## 📝 Advanced Features

### LoRA Management
- **Scan Directory**: Automatically finds LoRA models
- **Hide/Show**: Organize your model list
- **Scale Control**: Adjust style influence (0.0-3.0)

### Session Analytics
- Track images generated
- Monitor average generation time
- Calculate images per minute
- View GPU/memory usage

### Batch Metadata
Generated batches include:
- `batch_metadata.json` - Complete generation details
- Individual images with prompt in filename
- Seeds used for reproducibility

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

**Note**: FLUX.2-dev model has its own license from Black Forest Labs. Please review at:
https://huggingface.co/black-forest-labs/FLUX.2-dev

## 🙏 Acknowledgments

- **Black Forest Labs** - For FLUX.2-dev model
- **HuggingFace** - For Diffusers library
- **Gradio** - For UI framework
- **Community** - For testing and feedback

## 📞 Support

- **Issues**: https://github.com/Arifuzzamanjoy/Flux.2-dev-gradio-ui-runpod-A40-48-vram--H200-vram-/issues
- **Email**: s1710374103@ru.ac.bd

## 🔗 Links

- **FLUX.2-dev Model**: https://huggingface.co/black-forest-labs/FLUX.2-dev
- **4-bit Quantized**: https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit
- **Diffusers Docs**: https://huggingface.co/docs/diffusers
- **Gradio Docs**: https://www.gradio.app/docs

---

**Made with ❤️ by Arifuzzamanjoy**

*For RunPod, A40, H100, H200 GPU deployments*
