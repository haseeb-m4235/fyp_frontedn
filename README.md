# Gen2Seg Image Segmentation Frontend

A local Streamlit-based frontend for the Gen2Seg image segmentation system. This prototype provides a complete UI workflow for uploading, validating, and processing microscopy images.

## Features

### ✅ Complete Upload & Validation Pipeline
- **File Format Support**: TIFF (.tif, .tiff), PNG (.png), JPEG (.jpg, .jpeg)
- **Whitelist-based Validation**: Only accepted formats are allowed
- **Comprehensive Metadata Extraction**:
  - Image dimensions (2D/3D)
  - Channels, bit depth, data type
  - Physical pixel spacing (when available)
  - File size and format details
- **Dimensional Validation**:
  - 2D: 364×364 to 2048×2048 pixels
  - 3D: Up to 2048×2048×256 with minimum 3 slices
- **Secure Local Storage**: UUID-based upload directories with sanitized filenames

### ✅ Progress Tracking & Job Management
- Real-time progress bars with stage tracking
- Mock processing simulation with realistic timing
- Job status tracking (Created → Running → Completed)
- Unique Job IDs for each segmentation run

### ✅ Gen2Seg API Placeholder
- Clearly separated stub function ready for real implementation
- Simulates complete inference pipeline:
  - Image loading
  - Preprocessing
  - Model inference
  - Post-processing
  - Results saving

### ✅ Clean User Experience
- Intuitive step-by-step workflow
- Clear error messages and validation feedback
- Expandable sections for metadata and logs
- Status badges and visual feedback
- Responsive layout with sidebar controls

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**:
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

## Usage

### Step 1: Upload Image
1. Click "Choose an image file" button
2. Select a valid image file (.tif, .tiff, .png, .jpg, .jpeg)
3. Wait for automatic validation to complete

### Step 2: Review Metadata
- Examine extracted image properties
- Verify dimensions and file information
- Check validation results

### Step 3: Run Segmentation
1. Click "Run Segmentation" button
2. Monitor progress through processing stages
3. Wait for completion

### Step 4: View Results
- Review detected object count
- Check processing time and model version
- Access output files in `outputs/<job-id>/`

### Step 5: Process More Images
- Click "Process Another Image" to start over
- Or use "Clear All Data" in sidebar to reset everything

## File Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── uploads/              # Upload directory (auto-created)
│   └── <uuid>/          # Each upload gets unique directory
└── outputs/              # Output directory (auto-created)
    └── <job-id>/        # Each job gets unique directory
```

## Configuration

Edit constants in `app.py` to adjust limits:

```python
# File size limit
MAX_FILE_SIZE_MB = 500

# 2D image constraints
MIN_2D_SIZE = 364
MAX_2D_WIDTH = 2048
MAX_2D_HEIGHT = 2048

# 3D volume constraints
MAX_3D_WIDTH = 2048
MAX_3D_HEIGHT = 2048
MAX_3D_DEPTH = 256
MIN_3D_SLICES = 3
```

## Replacing the Mock API

The Gen2Seg inference is currently mocked in the `run_gen2seg_inference()` function. To integrate the real model:

1. **Locate the placeholder function** in `app.py`:
   ```python
   def run_gen2seg_inference(image_path: Path, output_dir: Path, progress_callback=None) -> Dict:
   ```

2. **Replace with your real implementation**:
   - Load the actual Gen2Seg model
   - Process the image at `image_path`
   - Save results to `output_dir`
   - Call `progress_callback(stage_name, progress)` for UI updates
   - Return a dict with actual results

3. **Expected return format**:
   ```python
   {
       'status': 'success',
       'mask_path': str,        # Path to segmentation mask
       'overlay_path': str,     # Path to overlay image
       'num_objects': int,      # Number of detected objects
       'processing_time': float,
       'model_version': str
   }
   ```

## Validation Flow

The app performs validation in this order:

1. ✅ **File Extension** - Whitelist check
2. ✅ **File Size** - Maximum 500 MB
3. ✅ **File Readability** - Can be opened by PIL/tifffile
4. ✅ **Metadata Extraction** - Successful parsing
5. ✅ **Dimensions** - Within allowed ranges

Any failure stops the process and displays an error.

## Security Features

- **Filename Sanitization**: Removes dangerous characters
- **Directory Isolation**: Each upload in unique UUID directory
- **No External Connections**: Fully offline operation
- **Local Storage Only**: No cloud uploads

## Known Limitations

- **HTTPS**: Not available for local Streamlit (HTTP only)
- **TIFF Support**: Requires `tifffile` package for full 3D support
- **Mock Processing**: Segmentation is currently simulated
- **No Authentication**: Designed for local single-user use

## Troubleshooting

### Issue: tifffile warning on startup
**Solution**: Install tifffile for full TIFF support:
```bash
pip install tifffile
```

### Issue: App won't start
**Solution**: Check Python version (3.8+ required) and dependencies:
```bash
python --version
pip install -r requirements.txt
```

### Issue: Files not saving
**Solution**: Check write permissions in the app directory

### Issue: Large 3D volumes crash the app
**Solution**: Reduce `MAX_3D_DEPTH` or implement streaming for large volumes

## Development Notes

### Code Structure
- **Modular Functions**: Each feature is self-contained
- **Clear Constants**: All limits defined at top of file
- **Session State**: Streamlit state management for workflow
- **Error Handling**: Try-catch blocks with user-friendly messages

### Extending the App
- Add more file format support in `ALLOWED_EXTENSIONS`
- Implement real-time preview in `render_metadata_display()`
- Add export options (ZIP download, etc.)
- Integrate with database for job history
- Add multi-file batch processing

## License

This is a prototype frontend for the Gen2Seg system. Adjust licensing as needed for your project.

## Support

For issues or questions about:
- **Frontend/UI**: Check this README and code comments
- **Gen2Seg Model**: Contact the model development team
- **Streamlit**: See [Streamlit Documentation](https://docs.streamlit.io)
