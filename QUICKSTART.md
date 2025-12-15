# Quick Start Guide

## Installation & Running (3 steps)

### Option 1: Automated (Linux/Mac)
```bash
./run.sh
```

### Option 2: Automated (Windows)
```cmd
run.bat
```

### Option 3: Manual
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## Generate Test Images

Create sample images for testing:

```bash
python generate_test_images.py
```

This creates a `test_images/` folder with valid and invalid test cases.

---

## Basic Usage

1. **Upload** - Drag and drop an image file (TIFF, PNG, or JPEG)
2. **Validate** - Wait for automatic validation (format, size, dimensions)
3. **Review** - Check extracted metadata
4. **Segment** - Click "Run Segmentation" button
5. **Results** - View mock segmentation results

---

## File Structure

```
project/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── generate_test_images.py     # Test image generator
├── run.sh / run.bat           # Quick start scripts
├── uploads/                    # Upload directory (auto-created)
└── outputs/                    # Results directory (auto-created)
```

---

## Key Features

✅ **Format Validation** - Only accepts .tif, .tiff, .png, .jpg, .jpeg
✅ **Size Validation** - Max 500 MB
✅ **Dimension Validation** - 2D: 364×364 to 2048×2048, 3D: up to 2048×2048×256
✅ **Metadata Extraction** - Dimensions, channels, bit depth, spacing
✅ **Progress Tracking** - Real-time progress bars
✅ **Job Management** - UUID-based upload and job IDs
✅ **Mock Segmentation** - Placeholder API ready for real implementation

---

## Next Steps

- **Replace Mock API**: Edit `run_gen2seg_inference()` in `app.py`
- **Add More Formats**: Update `ALLOWED_EXTENSIONS` constant
- **Adjust Limits**: Modify size/dimension constants
- **Customize UI**: Edit Streamlit components

For full documentation, see [README.md](README.md)
