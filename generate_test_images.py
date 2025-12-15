"""
Generate Test Images for Gen2Seg Frontend
This script creates sample images for testing the upload and validation system.
"""

import numpy as np
from PIL import Image
from pathlib import Path

# Create test_images directory
TEST_DIR = Path("test_images")
TEST_DIR.mkdir(exist_ok=True)

print("🔬 Generating test images for Gen2Seg frontend...")
print(f"Output directory: {TEST_DIR}/")
print()

# ============================================================================
# 1. Valid 2D Image (512x512)
# ============================================================================
print("📸 Creating valid 2D image (512×512, 8-bit)...")
img_2d = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
Image.fromarray(img_2d).save(TEST_DIR / "valid_2d_512x512.png")
print("   ✅ Saved: valid_2d_512x512.png")

# ============================================================================
# 2. Valid 2D RGB Image (1024x1024)
# ============================================================================
print("📸 Creating valid 2D RGB image (1024×1024, 8-bit)...")
img_2d_rgb = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
Image.fromarray(img_2d_rgb, mode='RGB').save(TEST_DIR / "valid_2d_1024x1024_rgb.jpg")
print("   ✅ Saved: valid_2d_1024x1024_rgb.jpg")

# ============================================================================
# 3. Valid 16-bit Image
# ============================================================================
print("📸 Creating valid 16-bit image (768×768)...")
img_16bit = np.random.randint(0, 65536, (768, 768), dtype=np.uint16)
Image.fromarray(img_16bit).save(TEST_DIR / "valid_16bit_768x768.tif")
print("   ✅ Saved: valid_16bit_768x768.tif")

# ============================================================================
# 4. Minimum Valid Size (364x364)
# ============================================================================
print("📸 Creating minimum valid size image (364×364)...")
img_min = np.random.randint(0, 256, (364, 364), dtype=np.uint8)
Image.fromarray(img_min).save(TEST_DIR / "valid_min_364x364.png")
print("   ✅ Saved: valid_min_364x364.png")

# ============================================================================
# 5. Maximum Valid Size (2048x2048)
# ============================================================================
print("📸 Creating maximum valid size image (2048×2048)...")
img_max = np.random.randint(0, 256, (2048, 2048), dtype=np.uint8)
Image.fromarray(img_max).save(TEST_DIR / "valid_max_2048x2048.png")
print("   ✅ Saved: valid_max_2048x2048.png")

# ============================================================================
# 6. 3D TIFF Stack (valid)
# ============================================================================
try:
    import tifffile
    print("📸 Creating valid 3D TIFF stack (512×512×10)...")
    img_3d = np.random.randint(0, 256, (10, 512, 512), dtype=np.uint8)
    tifffile.imwrite(TEST_DIR / "valid_3d_512x512x10.tif", img_3d)
    print("   ✅ Saved: valid_3d_512x512x10.tif")

    print("📸 Creating valid large 3D TIFF stack (1024×1024×50)...")
    img_3d_large = np.random.randint(0, 256, (50, 1024, 1024), dtype=np.uint8)
    tifffile.imwrite(TEST_DIR / "valid_3d_1024x1024x50.tif", img_3d_large)
    print("   ✅ Saved: valid_3d_1024x1024x50.tif")
except ImportError:
    print("   ⚠️  Skipping 3D images (tifffile not installed)")
    print("   Install with: pip install tifffile")

# ============================================================================
# 7. Invalid: Too Small (300x300)
# ============================================================================
print("📸 Creating INVALID image - too small (300×300)...")
img_small = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
Image.fromarray(img_small).save(TEST_DIR / "invalid_too_small_300x300.png")
print("   ❌ Saved: invalid_too_small_300x300.png (should be rejected)")

# ============================================================================
# 8. Invalid: Too Large (2500x2500)
# ============================================================================
print("📸 Creating INVALID image - too large (2500×2500)...")
img_large = np.random.randint(0, 256, (2500, 2500), dtype=np.uint8)
Image.fromarray(img_large).save(TEST_DIR / "invalid_too_large_2500x2500.png")
print("   ❌ Saved: invalid_too_large_2500x2500.png (should be rejected)")

# ============================================================================
# 9. Simulated Microscopy Image
# ============================================================================
print("📸 Creating simulated microscopy image with cells...")
# Create a more realistic looking microscopy image
img_micro = np.zeros((800, 800), dtype=np.uint8)
# Add some cell-like structures
for _ in range(20):
    x, y = np.random.randint(50, 750, 2)
    radius = np.random.randint(20, 50)
    y_coords, x_coords = np.ogrid[:800, :800]
    mask = (x_coords - x)**2 + (y_coords - y)**2 <= radius**2
    img_micro[mask] = np.random.randint(150, 255)
# Add some noise
noise = np.random.randint(0, 30, (800, 800), dtype=np.uint8)
img_micro = np.clip(img_micro + noise, 0, 255).astype(np.uint8)
Image.fromarray(img_micro).save(TEST_DIR / "simulated_microscopy_800x800.tif")
print("   ✅ Saved: simulated_microscopy_800x800.tif")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 60)
print("✅ Test image generation complete!")
print(f"📁 All images saved to: {TEST_DIR.absolute()}/")
print()
print("Test Cases:")
print("  ✅ Valid 2D images (various sizes and formats)")
print("  ✅ Valid 16-bit TIFF")
print("  ✅ Valid 3D TIFF stacks (if tifffile installed)")
print("  ✅ Minimum and maximum valid sizes")
print("  ✅ Simulated microscopy image")
print("  ❌ Invalid images (too small, too large)")
print()
print("Use these images to test the Gen2Seg frontend validation!")
print("=" * 60)
