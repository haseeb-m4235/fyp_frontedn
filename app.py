"""
Image Segmentation Frontend (Streamlit)
A local frontend for Gen2Seg image segmentation system.
"""

import streamlit as st
import os
import uuid
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import io

# Import image handling libraries
try:
    from PIL import Image
    import numpy as np
except ImportError:
    st.error("Required packages missing. Install: pip install pillow numpy tifffile")
    st.stop()

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    st.warning("tifffile not installed. TIFF support will be limited.")

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

ALLOWED_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Dimension constraints
MIN_2D_SIZE = 364
MAX_2D_WIDTH = 2048
MAX_2D_HEIGHT = 2048
MAX_3D_WIDTH = 2048
MAX_3D_HEIGHT = 2048
MAX_3D_DEPTH = 256
MIN_3D_SLICES = 3

# Directory structure
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other security issues."""
    # Keep only alphanumeric, dots, underscores, and hyphens
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '._-':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    return ''.join(safe_chars)


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """Validate file extension against whitelist."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    return True, "Valid file extension"


def validate_file_size(file_size: int) -> Tuple[bool, str]:
    """Validate file size is within limits."""
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({format_bytes(file_size)}) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
    return True, f"File size valid: {format_bytes(file_size)}"


def extract_image_metadata(file_path: Path, original_filename: str) -> Optional[Dict]:
    """
    Extract comprehensive metadata from uploaded image.
    Supports 2D and 3D images (TIFF stacks).
    """
    try:
        ext = file_path.suffix.lower()
        metadata = {
            'filename': original_filename,
            'file_size': file_path.stat().st_size,
            'file_size_formatted': format_bytes(file_path.stat().st_size),
        }

        # Handle TIFF files (may be 3D)
        if ext in ['.tif', '.tiff'] and TIFFFILE_AVAILABLE:
            with tifffile.TiffFile(file_path) as tif:
                image_data = tif.asarray()

                # Determine if 2D or 3D
                if image_data.ndim == 2:
                    metadata['dimensions'] = '2D'
                    metadata['width'] = image_data.shape[1]
                    metadata['height'] = image_data.shape[0]
                    metadata['depth'] = 1
                    metadata['channels'] = 1
                elif image_data.ndim == 3:
                    # Could be 3D grayscale or 2D RGB
                    if image_data.shape[2] <= 4:  # Likely RGB/RGBA
                        metadata['dimensions'] = '2D'
                        metadata['width'] = image_data.shape[1]
                        metadata['height'] = image_data.shape[0]
                        metadata['depth'] = 1
                        metadata['channels'] = image_data.shape[2]
                    else:  # Likely 3D stack
                        metadata['dimensions'] = '3D'
                        metadata['depth'] = image_data.shape[0]
                        metadata['height'] = image_data.shape[1]
                        metadata['width'] = image_data.shape[2]
                        metadata['channels'] = 1
                elif image_data.ndim == 4:  # 3D with channels
                    metadata['dimensions'] = '3D'
                    metadata['depth'] = image_data.shape[0]
                    metadata['height'] = image_data.shape[1]
                    metadata['width'] = image_data.shape[2]
                    metadata['channels'] = image_data.shape[3]
                else:
                    return None

                metadata['dtype'] = str(image_data.dtype)
                metadata['bit_depth'] = image_data.dtype.itemsize * 8

                # Try to extract physical spacing from TIFF tags
                try:
                    for page in tif.pages:
                        tags = page.tags
                        if 'XResolution' in tags and 'YResolution' in tags:
                            x_res = tags['XResolution'].value
                            y_res = tags['YResolution'].value
                            if isinstance(x_res, tuple):
                                x_res = x_res[0] / x_res[1] if x_res[1] != 0 else None
                            if isinstance(y_res, tuple):
                                y_res = y_res[0] / y_res[1] if y_res[1] != 0 else None
                            metadata['physical_spacing_x'] = x_res
                            metadata['physical_spacing_y'] = y_res
                        break
                except:
                    pass

        # Handle standard image formats (PNG, JPG)
        else:
            with Image.open(file_path) as img:
                metadata['dimensions'] = '2D'
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['depth'] = 1
                metadata['channels'] = len(img.getbands())
                metadata['mode'] = img.mode

                # Get bit depth
                if img.mode == 'L':
                    metadata['bit_depth'] = 8
                    metadata['dtype'] = 'uint8'
                elif img.mode == 'I':
                    metadata['bit_depth'] = 32
                    metadata['dtype'] = 'int32'
                elif img.mode == 'F':
                    metadata['bit_depth'] = 32
                    metadata['dtype'] = 'float32'
                elif img.mode in ['RGB', 'RGBA']:
                    metadata['bit_depth'] = 8
                    metadata['dtype'] = 'uint8'
                elif img.mode == 'I;16':
                    metadata['bit_depth'] = 16
                    metadata['dtype'] = 'uint16'
                else:
                    metadata['bit_depth'] = 8
                    metadata['dtype'] = 'unknown'

                # Try to get DPI (physical spacing)
                dpi = img.info.get('dpi')
                if dpi:
                    metadata['physical_spacing_x'] = dpi[0]
                    metadata['physical_spacing_y'] = dpi[1]

        return metadata

    except Exception as e:
        st.error(f"Error extracting metadata: {str(e)}")
        return None


def validate_dimensions(metadata: Dict) -> Tuple[bool, str]:
    """Validate image dimensions meet requirements."""
    width = metadata['width']
    height = metadata['height']
    depth = metadata.get('depth', 1)
    dimensions = metadata['dimensions']

    if dimensions == '2D':
        # Check minimum size
        if width < MIN_2D_SIZE or height < MIN_2D_SIZE:
            return False, f"2D image too small. Minimum: {MIN_2D_SIZE}×{MIN_2D_SIZE}. Got: {width}×{height}"

        # Check maximum size
        if width > MAX_2D_WIDTH or height > MAX_2D_HEIGHT:
            return False, f"2D image too large. Maximum: {MAX_2D_WIDTH}×{MAX_2D_HEIGHT}. Got: {width}×{height}"

        return True, f"2D dimensions valid: {width}×{height}"

    elif dimensions == '3D':
        # Check Z-dimension minimum
        if depth < MIN_3D_SLICES:
            return False, f"3D volume requires at least {MIN_3D_SLICES} slices. Got: {depth}"

        # Check maximum size
        if width > MAX_3D_WIDTH or height > MAX_3D_HEIGHT or depth > MAX_3D_DEPTH:
            return False, f"3D volume too large. Maximum: {MAX_3D_WIDTH}×{MAX_3D_HEIGHT}×{MAX_3D_DEPTH}. Got: {width}×{height}×{depth}"

        return True, f"3D dimensions valid: {width}×{height}×{depth}"

    return False, "Unknown dimension type"


# ============================================================================
# GEN2SEG API PLACEHOLDER
# ============================================================================

def run_gen2seg_inference(image_path: Path, output_dir: Path, progress_callback=None) -> Dict:
    """
    PLACEHOLDER for Gen2Seg segmentation model.
    This function will be replaced by the real API.

    Args:
        image_path: Path to input image
        output_dir: Directory to save segmentation results
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with segmentation results and paths
    """
    # Simulate processing stages
    stages = [
        ("Loading image into memory", 2),
        ("Preprocessing image data", 1.5),
        ("Loading Gen2Seg model", 2.5),
        ("Running inference on GPU", 5),
        ("Post-processing segmentation mask", 1.5),
        ("Saving results", 1)
    ]

    total_time = sum(s[1] for s in stages)
    elapsed = 0

    for stage_name, duration in stages:
        if progress_callback:
            progress_callback(stage_name, elapsed / total_time)
        time.sleep(duration)
        elapsed += duration

    if progress_callback:
        progress_callback("Completed", 1.0)

    # Create mock segmentation mask
    mask_path = output_dir / "segmentation_mask.png"
    overlay_path = output_dir / "segmentation_overlay.png"

    # Mock file creation
    mask_path.touch()
    overlay_path.touch()

    return {
        'status': 'success',
        'mask_path': str(mask_path),
        'overlay_path': str(overlay_path),
        'num_objects': 42,  # Mock number of detected objects
        'processing_time': total_time,
        'model_version': 'Gen2Seg-v1.0-MOCK'
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'upload_uuid' not in st.session_state:
        st.session_state.upload_uuid = None
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'validation_passed' not in st.session_state:
        st.session_state.validation_passed = False
    if 'job_id' not in st.session_state:
        st.session_state.job_id = None
    if 'job_status' not in st.session_state:
        st.session_state.job_status = None
    if 'segmentation_results' not in st.session_state:
        st.session_state.segmentation_results = None


def render_header():
    """Render application header."""
    st.title("🔬 Gen2Seg Image Segmentation")
    st.markdown("### Upload and segment microscopy images locally")
    st.divider()


def render_upload_instructions():
    """Render upload instructions and constraints."""
    with st.expander("📋 Upload Instructions & Requirements", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Accepted File Formats:**")
            st.markdown("- TIFF (`.tif`, `.tiff`)")
            st.markdown("- PNG (`.png`)")
            st.markdown("- JPEG (`.jpg`, `.jpeg`)")

            st.markdown("**File Size Limit:**")
            st.markdown(f"- Maximum: **{MAX_FILE_SIZE_MB} MB**")

        with col2:
            st.markdown("**Dimension Requirements (2D):**")
            st.markdown(f"- Minimum: {MIN_2D_SIZE}×{MIN_2D_SIZE} pixels")
            st.markdown(f"- Maximum: {MAX_2D_WIDTH}×{MAX_2D_HEIGHT} pixels")

            st.markdown("**Dimension Requirements (3D):**")
            st.markdown(f"- Z-slices: ≥{MIN_3D_SLICES}")
            st.markdown(f"- Maximum: {MAX_3D_WIDTH}×{MAX_3D_HEIGHT}×{MAX_3D_DEPTH}")


def render_file_upload():
    """Render file upload section with validation."""
    st.subheader("1️⃣ Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB"
    )

    if uploaded_file is not None:
        # Reset previous state when new file is uploaded
        if st.session_state.uploaded_file_path != uploaded_file.name:
            st.session_state.validation_passed = False
            st.session_state.metadata = None
            st.session_state.job_status = None
            st.session_state.segmentation_results = None

        st.info(f"📁 File selected: **{uploaded_file.name}** ({format_bytes(uploaded_file.size)})")

        # Validation process
        with st.spinner("Validating upload..."):
            validation_results = []

            # Step 1: Validate file extension
            ext_valid, ext_msg = validate_file_extension(uploaded_file.name)
            validation_results.append(("File Extension", ext_valid, ext_msg))

            if not ext_valid:
                st.error(f"❌ {ext_msg}")
                return

            # Step 2: Validate file size
            size_valid, size_msg = validate_file_size(uploaded_file.size)
            validation_results.append(("File Size", size_valid, size_msg))

            if not size_valid:
                st.error(f"❌ {size_msg}")
                return

            # Step 3: Save file locally and validate it can be opened
            try:
                # Generate UUID and create upload directory
                upload_uuid = str(uuid.uuid4())
                upload_path = UPLOAD_DIR / upload_uuid
                upload_path.mkdir(parents=True, exist_ok=True)

                # Sanitize filename and save
                safe_filename = sanitize_filename(uploaded_file.name)
                file_path = upload_path / safe_filename

                # Write file to disk
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                validation_results.append(("File Saved", True, f"Saved to {upload_uuid}/"))

            except Exception as e:
                st.error(f"❌ Error saving file: {str(e)}")
                return

            # Step 4: Extract metadata
            metadata = extract_image_metadata(file_path, uploaded_file.name)

            if metadata is None:
                st.error("❌ Could not read image file. File may be corrupted.")
                # Clean up
                shutil.rmtree(upload_path, ignore_errors=True)
                return

            validation_results.append(("Metadata Extraction", True, "Successfully extracted image metadata"))

            # Step 5: Validate dimensions
            dim_valid, dim_msg = validate_dimensions(metadata)
            validation_results.append(("Dimensions", dim_valid, dim_msg))

            if not dim_valid:
                st.error(f"❌ {dim_msg}")
                # Clean up
                shutil.rmtree(upload_path, ignore_errors=True)
                return

            # All validations passed
            st.session_state.upload_uuid = upload_uuid
            st.session_state.uploaded_file_path = str(file_path)
            st.session_state.metadata = metadata
            st.session_state.validation_passed = True

            # Display validation summary
            st.success("✅ All validations passed!")

            with st.expander("🔍 Validation Results", expanded=True):
                for check_name, passed, message in validation_results:
                    if passed:
                        st.markdown(f"✅ **{check_name}**: {message}")
                    else:
                        st.markdown(f"❌ **{check_name}**: {message}")


def render_metadata_display():
    """Display extracted image metadata."""
    if st.session_state.metadata is None:
        return

    st.subheader("2️⃣ Image Metadata")

    metadata = st.session_state.metadata

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Upload ID", st.session_state.upload_uuid[:8] + "...")
        st.metric("Dimensions", metadata['dimensions'])
        st.metric("File Size", metadata['file_size_formatted'])

    with col2:
        if metadata['dimensions'] == '2D':
            st.metric("Width × Height", f"{metadata['width']} × {metadata['height']}")
        else:
            st.metric("Width × Height × Depth",
                     f"{metadata['width']} × {metadata['height']} × {metadata['depth']}")
        st.metric("Channels", metadata['channels'])

    with col3:
        st.metric("Data Type", metadata['dtype'])
        st.metric("Bit Depth", f"{metadata['bit_depth']}-bit")

    # Additional metadata in expander
    with st.expander("📊 Detailed Metadata"):
        st.json(metadata)


def render_segmentation_control():
    """Render segmentation job control section."""
    if not st.session_state.validation_passed:
        st.info("⏳ Complete file upload and validation to enable segmentation.")
        return

    st.subheader("3️⃣ Run Segmentation")

    # Check if job is already running or completed
    if st.session_state.job_status == 'running':
        st.warning("⚙️ Segmentation job is currently running...")
        return

    if st.session_state.job_status == 'completed':
        st.success("✅ Segmentation completed!")

        if st.button("🔄 Run New Segmentation", type="secondary"):
            st.session_state.job_status = None
            st.session_state.segmentation_results = None
            st.rerun()
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("🚀 Ready to segment! Click the button to start processing.")

    with col2:
        if st.button("▶️ Run Segmentation", type="primary", use_container_width=True):
            # Create segmentation job
            job_id = str(uuid.uuid4())
            st.session_state.job_id = job_id
            st.session_state.job_status = 'running'
            st.rerun()


def render_segmentation_progress():
    """Render segmentation progress and results."""
    if st.session_state.job_status != 'running':
        return

    st.subheader("4️⃣ Segmentation Progress")

    # Create output directory
    output_path = OUTPUT_DIR / st.session_state.job_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Progress tracking UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Job info
    with st.expander("📋 Job Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Job ID:** `{st.session_state.job_id[:16]}...`")
            st.markdown(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.markdown(f"**Status:** 🟡 Running")
            st.markdown(f"**Image:** {st.session_state.metadata['filename']}")

    # Callback for progress updates
    def progress_callback(stage: str, progress: float):
        progress_bar.progress(progress)
        status_text.markdown(f"**Current Stage:** {stage} ({int(progress * 100)}%)")

    # Run Gen2Seg inference (mocked)
    try:
        results = run_gen2seg_inference(
            Path(st.session_state.uploaded_file_path),
            output_path,
            progress_callback
        )

        # Update session state
        st.session_state.job_status = 'completed'
        st.session_state.segmentation_results = results

        progress_bar.progress(1.0)
        status_text.markdown("**Status:** ✅ Completed")

        st.success("🎉 Segmentation completed successfully!")
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Segmentation failed: {str(e)}")
        st.session_state.job_status = 'failed'


def render_results():
    """Render segmentation results."""
    if st.session_state.segmentation_results is None:
        return

    st.subheader("5️⃣ Segmentation Results")

    results = st.session_state.segmentation_results

    # Results summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Objects Detected", results['num_objects'])

    with col2:
        st.metric("Processing Time", f"{results['processing_time']:.1f}s")

    with col3:
        st.metric("Model Version", results['model_version'])

    # Output files
    with st.expander("📁 Output Files", expanded=True):
        st.markdown(f"**Job ID:** `{st.session_state.job_id}`")
        st.markdown(f"**Output Directory:** `outputs/{st.session_state.job_id}/`")
        st.markdown("---")
        st.markdown(f"- 🎭 Segmentation Mask: `{Path(results['mask_path']).name}`")
        st.markdown(f"- 🖼️ Overlay Image: `{Path(results['overlay_path']).name}`")

        st.info("💡 Output files are saved locally and ready for download or further processing.")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 View Output Directory", use_container_width=True):
            st.code(f"outputs/{st.session_state.job_id}/", language="bash")

    with col2:
        if st.button("🔄 Process Another Image", type="primary", use_container_width=True):
            # Reset state
            st.session_state.upload_uuid = None
            st.session_state.uploaded_file_path = None
            st.session_state.metadata = None
            st.session_state.validation_passed = False
            st.session_state.job_id = None
            st.session_state.job_status = None
            st.session_state.segmentation_results = None
            st.rerun()


def render_sidebar():
    """Render sidebar with system information and controls."""
    with st.sidebar:
        st.header("⚙️ System Information")

        st.markdown("### 📊 Storage")
        st.markdown(f"- **Uploads:** `{UPLOAD_DIR}/`")
        st.markdown(f"- **Outputs:** `{OUTPUT_DIR}/`")

        st.divider()

        st.markdown("### 🔧 Configuration")
        st.markdown(f"- Max File Size: {MAX_FILE_SIZE_MB} MB")
        st.markdown(f"- 2D Min: {MIN_2D_SIZE}×{MIN_2D_SIZE}")
        st.markdown(f"- 2D Max: {MAX_2D_WIDTH}×{MAX_2D_HEIGHT}")
        st.markdown(f"- 3D Max: {MAX_3D_WIDTH}×{MAX_3D_HEIGHT}×{MAX_3D_DEPTH}")

        st.divider()

        st.markdown("### 📦 Dependencies")
        st.markdown(f"- Streamlit {st.__version__}")
        st.markdown(f"- PIL (Pillow)")
        st.markdown(f"- NumPy")
        if TIFFFILE_AVAILABLE:
            st.markdown("- tifffile ✅")
        else:
            st.markdown("- tifffile ⚠️ (not installed)")

        st.divider()

        if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
            # Cleanup directories
            if st.confirm("Are you sure? This will delete all uploads and outputs."):
                shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
                shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
                UPLOAD_DIR.mkdir(exist_ok=True)
                OUTPUT_DIR.mkdir(exist_ok=True)

                # Reset session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                st.success("✅ All data cleared!")
                st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Gen2Seg Image Segmentation",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        .stAlert {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Render main content
    render_header()
    render_upload_instructions()
    render_file_upload()
    render_metadata_display()
    render_segmentation_control()
    render_segmentation_progress()
    render_results()

    # Footer
    st.divider()
    st.markdown(
        "**Note:** This is a local frontend prototype. "
        "The Gen2Seg model is currently mocked and will be replaced with the real implementation."
    )


if __name__ == "__main__":
    main()
