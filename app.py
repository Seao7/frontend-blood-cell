# frontend/app.py - Optimized for Streamlit Cloud + Kaggle GPU Service
import streamlit as st
import cv2
import numpy as np
import requests
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import tempfile
import os

# SciPy ecosystem imports
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import distance_matrix

# Scikit-image imports
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

st.set_page_config(
    page_title="Blood Cell Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Blood Cell Analysis Platform - Powered by SAM and Computer Vision"
    }
)

class BloodCellAnalyzer:
    """Complete blood cell analysis logic - runs on CPU in frontend"""
    
    def __init__(self):
        self.average_cell_area = 800
        self.cluster_density_factor = 0.7
    
    def create_circular_mask_simple(self, image_gray, threshold=15):
        """Create circular mask for blood sample area"""
        mask = image_gray > threshold
        mask = ndimage.binary_fill_holes(mask)
        
        print(f"✅ Circular mask created: {np.sum(mask)} valid pixels ({np.sum(mask)/mask.size*100:.1f}%)")
        return mask

    def apply_mask_to_detections(self, masks, circular_mask, min_overlap=0.5):
        """Filter masks that don't overlap sufficiently with circular area"""
        filtered_masks = []
        removed = 0
        
        for mask in masks:
            overlap = mask & circular_mask
            overlap_ratio = np.sum(overlap) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            if overlap_ratio >= min_overlap:
                cleaned_mask = mask & circular_mask
                if np.sum(cleaned_mask) > 50:  
                    filtered_masks.append(cleaned_mask)
                else:
                    removed += 1
            else:
                removed += 1
        
        print(f"🚫 Removed {removed} detections outside circular area")
        return filtered_masks

    def extract_centroids(self, masks):
        """Extract centroids from masks"""
        centroids = []
        for mask in masks:
            y, x = np.where(mask)
            if len(y) > 0:
                centroids.append((np.mean(y), np.mean(x)))
        return centroids

    def compute_circularity_heatmap(self, masks, shape, strictness=2.0):
        """Compute circularity heatmap for individual detected cells"""
        circ_map = np.zeros(shape, dtype=np.float32)
        scores = []

        for mask in masks:
            props = measure.regionprops(mask.astype(int))[0]
            area, perimeter = props.area, props.perimeter
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                circularity = min(circularity, 1.0) ** strictness
            else:
                circularity = 0
            scores.append(circularity)
            circ_map[mask] = circularity

        return gaussian_filter(circ_map, sigma=3), scores

    def detect_and_split_merged_cells(self, masks, min_concavity_ratio=0.1):
        """Split merged cells using watershed algorithm"""
        split_masks = []
        concave_count = 0
        non_concave_count = 0

        for mask in masks:
            props = measure.regionprops(mask.astype(int))[0]
            if props.solidity < (1 - min_concavity_ratio):
                concave_count += 1
                results = self.split_concave_cell(mask)
                split_masks.extend(results if len(results) > 1 else [mask])
            else:
                non_concave_count += 1
                split_masks.append(mask)

        print(f"🟠 Solidity-based merged candidates: {concave_count}")
        print(f"🟢 Non-concave cells (unchanged): {non_concave_count}")
        print(f"🧩 Final cell masks after split: {len(split_masks)}")

        return split_masks

    def split_concave_cell(self, mask):
        """Split concave cells using watershed"""
        distance = ndimage.distance_transform_edt(mask)
        local_max = peak_local_max(distance, min_distance=15, threshold_abs=0.3 * np.max(distance))

        if len(local_max) < 2:
            return [mask]

        markers = np.zeros_like(distance, dtype=int)
        for i, (y, x) in enumerate(local_max):
            markers[y, x] = i + 1

        labels = watershed(-distance, markers, mask=mask)
        return [labels == i for i in np.unique(labels) if i != 0 and np.sum(labels == i) > 50]

    def create_combined_cell_mask(self, cell_masks):
        """Combine all individual cell masks into a single mask"""
        if not cell_masks:
            return None

        combined_mask = np.zeros_like(cell_masks[0], dtype=bool)
        for cell_mask in cell_masks:
            combined_mask = combined_mask | cell_mask

        return combined_mask

    def is_cluster_like_shape(self, mask, max_aspect_ratio=3.0, min_solidity=0.7,
                              min_extent=0.3, min_compactness=0.2):
        """Determine if a mask has cluster-like geometric properties"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, {}

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area == 0:
            return False, {}

        # Calculate shape metrics
        rect = cv2.minAreaRect(largest_contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        x, y, w, h = cv2.boundingRect(largest_contour)
        extent = area / (w * h) if (w * h) > 0 else 0

        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        # Check criteria
        is_cluster = (aspect_ratio <= max_aspect_ratio and
                     solidity >= min_solidity and
                     extent >= min_extent and
                     compactness >= min_compactness)

        metrics = {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent,
            'compactness': compactness
        }

        return is_cluster, metrics

    def filter_clusters_by_overlap_and_shape(self, cluster_masks, cell_masks,
                                           max_overlap_percentage=10.0,
                                           max_aspect_ratio=3.0, min_solidity=0.7,
                                           min_extent=0.3, min_compactness=0.2):
        """Filter clusters by both overlap with cells and shape criteria"""
        if not cluster_masks:
            return [], []

        combined_cell_mask = self.create_combined_cell_mask(cell_masks) if cell_masks else None
        valid_clusters = []
        cluster_details = []

        for i, cluster_mask in enumerate(cluster_masks):
            # Check shape criteria
            is_cluster_shape, shape_metrics = self.is_cluster_like_shape(
                cluster_mask, max_aspect_ratio, min_solidity, min_extent, min_compactness
            )

            # Check overlap criteria
            overlap_ok = True
            overlap_percentage = 0
            if combined_cell_mask is not None:
                overlap = cluster_mask & combined_cell_mask
                overlap_area = np.sum(overlap)
                cluster_area = np.sum(cluster_mask)
                overlap_percentage = (overlap_area / cluster_area * 100) if cluster_area > 0 else 0
                overlap_ok = overlap_percentage <= max_overlap_percentage

            # Keep cluster if it passes both criteria
            if is_cluster_shape and overlap_ok:
                valid_clusters.append(cluster_mask)
                print(f"Cluster {i}: KEPT (area: {shape_metrics['area']:.0f}, overlap: {overlap_percentage:.1f}%)")
            else:
                reasons = []
                if not is_cluster_shape:
                    reasons.append("shape")
                if not overlap_ok:
                    reasons.append(f"overlap ({overlap_percentage:.1f}%)")
                print(f"Cluster {i}: REMOVED ({', '.join(reasons)})")

            cluster_details.append({
                'cluster_id': i,
                'area': shape_metrics.get('area', 0),
                'overlap_percentage': overlap_percentage,
                'is_valid': is_cluster_shape and overlap_ok,
                'shape_metrics': shape_metrics
            })

        return valid_clusters, cluster_details

    def estimate_cells_in_cluster(self, cluster_mask, average_cell_area=None):
        """Estimate number of cells in a cluster based on area and density"""
        if average_cell_area is None:
            average_cell_area = self.average_cell_area

        cluster_area = np.sum(cluster_mask)

        # Estimate cells considering packing density
        estimated_cells = max(1, int((cluster_area * self.cluster_density_factor) / average_cell_area))

        return estimated_cells

    def generate_estimated_cell_positions(self, cluster_mask, num_cells):
        """Generate estimated cell positions within a cluster using grid-based approach"""
        if num_cells <= 1:
            # Return cluster centroid
            y, x = np.where(cluster_mask)
            return [(np.mean(y), np.mean(x))] if len(y) > 0 else []

        # Get cluster boundary
        y_coords, x_coords = np.where(cluster_mask)
        if len(y_coords) == 0:
            return []

        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)

        # Create grid of potential positions
        grid_size = int(np.sqrt(num_cells)) + 1
        y_positions = np.linspace(min_y, max_y, grid_size)
        x_positions = np.linspace(min_x, max_x, grid_size)

        estimated_positions = []
        positions_added = 0

        for y_pos in y_positions:
            for x_pos in x_positions:
                if positions_added >= num_cells:
                    break

                # Check if position is within cluster
                y_idx, x_idx = int(y_pos), int(x_pos)
                if (0 <= y_idx < cluster_mask.shape[0] and
                    0 <= x_idx < cluster_mask.shape[1] and
                    cluster_mask[y_idx, x_idx]):
                    estimated_positions.append((y_pos, x_pos))
                    positions_added += 1

            if positions_added >= num_cells:
                break

        # If we don't have enough positions, add some random ones within the cluster
        while len(estimated_positions) < num_cells and len(y_coords) > 0:
            idx = np.random.randint(len(y_coords))
            estimated_positions.append((y_coords[idx], x_coords[idx]))

        return estimated_positions

    def create_mini_mask_at_position(self, pos, cluster_mask):
        """Create small circular mask around position"""
        y, x = int(pos[0]), int(pos[1])
        mini_mask = np.zeros_like(cluster_mask)
        radius = int(np.sqrt(self.average_cell_area / np.pi))
        y_min, y_max = max(0, y-radius), min(cluster_mask.shape[0], y+radius)
        x_min, x_max = max(0, x-radius), min(cluster_mask.shape[1], x+radius)

        for dy in range(y_min, y_max):
            for dx in range(x_min, x_max):
                if (dy - y)**2 + (dx - x)**2 <= radius**2:
                    mini_mask[dy, dx] = True

        # Mask to only cluster area
        mini_mask = mini_mask & cluster_mask
        return mini_mask

    def compute_unified_aggregation_map(self, all_centroids, all_masks, shape, k=5, sigma=3):
        """Compute aggregation map using both detected cells and estimated cluster cells"""
        if len(all_centroids) < 2:
            return np.zeros(shape, dtype=np.float32), []

        dmat = distance_matrix(all_centroids, all_centroids)
        np.fill_diagonal(dmat, np.inf)

        mean_dists = np.mean(np.sort(dmat, axis=1)[:, :k], axis=1)
        agg_scores = mean_dists

        heatmap = np.zeros(shape, dtype=np.float32)
        for mask, score in zip(all_masks, agg_scores):
            heatmap[mask] = score/max(agg_scores) if max(agg_scores) > 0 else 0

        heatmap = gaussian_filter(heatmap, sigma=sigma)
        return heatmap, agg_scores
    
    def analyze_from_masks(self, image, cell_masks_data, cluster_masks_data):
        """Analyze blood cells from pre-generated SAM masks"""
        print("🔬 Starting Blood Cell Analysis from SAM masks")
        
        # Convert mask data back to numpy arrays
        cell_masks = [np.array(mask['segmentation'], dtype=bool) for mask in cell_masks_data]
        cluster_masks = [np.array(mask['segmentation'], dtype=bool) for mask in cluster_masks_data]
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply circular mask if needed
        circular_mask = self.create_circular_mask_simple(image_gray)
        if circular_mask is not None:
            cell_masks = self.apply_mask_to_detections(cell_masks, circular_mask)
            cluster_masks = self.apply_mask_to_detections(cluster_masks, circular_mask)
        
        # Split merged cells
        cell_masks = self.detect_and_split_merged_cells(cell_masks)
        
        # Filter clusters
        valid_clusters, cluster_details = self.filter_clusters_by_overlap_and_shape(
            cluster_masks, cell_masks
        )
        
        # Extract centroids
        cell_centroids = self.extract_centroids(cell_masks)
        
        # Compute circularity for individual cells
        circularity_heatmap, circularity_scores = self.compute_circularity_heatmap(
            cell_masks, image_gray.shape
        )
        
        # Update average cell area
        if cell_masks:
            cell_areas = [np.sum(mask) for mask in cell_masks]
            self.average_cell_area = np.mean(cell_areas)
        
        # Estimate cells in clusters
        estimated_cell_info = []
        all_estimated_centroids = []
        all_estimated_masks = []
        total_estimated_cells = 0
        
        for i, cluster_mask in enumerate(valid_clusters):
            estimated_count = self.estimate_cells_in_cluster(cluster_mask)
            estimated_positions = self.generate_estimated_cell_positions(cluster_mask, estimated_count)
            
            for pos in estimated_positions:
                mini_mask = self.create_mini_mask_at_position(pos, cluster_mask)
                if np.sum(mini_mask) > 0:
                    all_estimated_masks.append(mini_mask)
                    all_estimated_centroids.append(pos)
            
            total_estimated_cells += estimated_count
            estimated_cell_info.append({
                'cluster_id': i,
                'area': np.sum(cluster_mask),
                'estimated_cells': estimated_count,
                'positions': estimated_positions
            })
        
        # Unified aggregation calculation
        all_centroids = cell_centroids + all_estimated_centroids
        all_masks = cell_masks + all_estimated_masks
        
        unified_heatmap, agg_scores = self.compute_unified_aggregation_map(
            all_centroids, all_masks, image_gray.shape, k=5
        )
        
        total_cells = len(cell_masks) + total_estimated_cells
        
        # Print comprehensive report
        self.print_analysis_report(cell_masks, valid_clusters, estimated_cell_info, 
                                 circularity_scores, agg_scores, total_cells)
        
        return {
            'cell_masks': cell_masks,
            'cluster_masks': valid_clusters,
            'estimated_cell_info': estimated_cell_info,
            'unified_heatmap': unified_heatmap,
            'circularity_heatmap': circularity_heatmap,
            'aggregation_scores': agg_scores,
            'circularity_scores': circularity_scores,
            'total_cells': total_cells,
            'individual_cells': len(cell_masks),
            'estimated_cluster_cells': total_estimated_cells,
            'all_centroids': all_centroids
        }

    def print_analysis_report(self, cell_masks, valid_clusters, estimated_cell_info, 
                            circularity_scores, agg_scores, total_cells):
        """Print comprehensive analysis report"""
        print(f"\n" + "="*60)
        print(f"📊 COMPREHENSIVE ANALYSIS REPORT")
        print(f"="*60)

        # Individual Cell Detection Results
        print(f"\n🔴 INDIVIDUAL CELL DETECTION:")
        print(f"  ├─ Cells detected: {len(cell_masks)}")
        print(f"  ├─ Average cell area: {self.average_cell_area:.0f} pixels")
        if cell_masks:
            cell_areas = [np.sum(mask) for mask in cell_masks]
            print(f"  ├─ Cell area range: {min(cell_areas):.0f} - {max(cell_areas):.0f} pixels")
        print(f"  └─ Coverage: {len(cell_masks)/total_cells*100:.1f}% of total cells")

        # Cluster Detection Results
        print(f"\n🟢 CLUSTER DETECTION:")
        if valid_clusters:
            cluster_areas = [np.sum(mask) for mask in valid_clusters]
            total_estimated = sum(info['estimated_cells'] for info in estimated_cell_info)
            print(f"  ├─ Clusters found: {len(valid_clusters)}")
            print(f"  ├─ Average cluster area: {np.mean(cluster_areas):.0f} pixels")
            print(f"  ├─ Cluster area range: {min(cluster_areas):.0f} - {max(cluster_areas):.0f} pixels")
            print(f"  ├─ Total estimated cells: {total_estimated}")
            print(f"  └─ Coverage: {total_estimated/total_cells*100:.1f}% of total cells")
        else:
            print(f"  └─ No valid clusters detected")

        # Circularity Quality Metrics
        print(f"\n🔵 CELL SHAPE QUALITY (Circularity):")
        if circularity_scores:
            excellent_cells = sum(1 for score in circularity_scores if score >= 0.8)
            good_cells = sum(1 for score in circularity_scores if 0.6 <= score < 0.8)
            poor_cells = sum(1 for score in circularity_scores if score < 0.6)

            print(f"  ├─ Mean circularity: {np.mean(circularity_scores):.3f}")
            print(f"  ├─ Std deviation: {np.std(circularity_scores):.3f}")
            print(f"  ├─ Excellent (≥0.8): {excellent_cells} cells ({excellent_cells/len(circularity_scores)*100:.1f}%)")
            print(f"  ├─ Good (0.6-0.8): {good_cells} cells ({good_cells/len(circularity_scores)*100:.1f}%)")
            print(f"  └─ Poor (<0.6): {poor_cells} cells ({poor_cells/len(circularity_scores)*100:.1f}%)")

        # Aggregation/Density Metrics
        print(f"\n📏 CELL AGGREGATION ANALYSIS:")
        if len(agg_scores):
            high_agg = sum(1 for score in agg_scores if score < 30)
            med_agg = sum(1 for score in agg_scores if 30 <= score < 50)
            low_agg = sum(1 for score in agg_scores if score >= 50)

            print(f"  ├─ Mean aggregation score: {np.mean(agg_scores):.2f} pixels")
            print(f"  ├─ High density (<30px): {high_agg} cells ({high_agg/len(agg_scores)*100:.1f}%)")
            print(f"  ├─ Medium density (30-50px): {med_agg} cells ({med_agg/len(agg_scores)*100:.1f}%)")
            print(f"  └─ Low density (≥50px): {low_agg} cells ({low_agg/len(agg_scores)*100:.1f}%)")

        print(f"\n" + "="*60)

def extract_frames_from_video(video_bytes, fps=2, max_duration=5):
    """Extract frames from video - optimized for cloud deployment"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    
    frames = []
    frame_count = 0
    max_frames = fps * max_duration
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    os.unlink(tmp_path)
    return frames

def call_mask_service(frame):
    """Call GPU service - optimized for Kaggle tunneling"""
    # Convert frame to base64 with compression for faster upload
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85, optimize=True)  # Compress for Kaggle
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Kaggle + Streamlit Cloud URL handling
    GPU_SERVICE_URL = None
    
    # Try Streamlit secrets first (preferred for production)
    try:
        GPU_SERVICE_URL = st.secrets["GPU_SERVICE_URL"]
        st.success(f"🔗 Using Kaggle GPU service from secrets")
    except:
        # Fall back to environment variable
        GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL")
        if GPU_SERVICE_URL:
            st.info(f"🔗 Using GPU service from environment")
    
    if not GPU_SERVICE_URL:
        st.error("❌ **GPU Service URL not configured!**")
        st.error("Please configure GPU_SERVICE_URL in Streamlit Cloud secrets:")
        st.code("""
        # In Streamlit Cloud dashboard:
        # Settings → Secrets → Add:
        GPU_SERVICE_URL = "https://your-kaggle-ngrok-url.ngrok.io"
        """)
        return None
    
    # Validate URL format
    if not GPU_SERVICE_URL.startswith(('http://', 'https://')):
        st.error(f"❌ Invalid GPU service URL format: {GPU_SERVICE_URL}")
        return None
    
    payload = {"frame_data": img_b64}
    
    try:
        with st.spinner("🔥 Connecting to Kaggle GPU service..."):
            response = requests.post(
                f"{GPU_SERVICE_URL}/generate_masks", 
                json=payload, 
                timeout=180,  # Longer timeout for Kaggle
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
        
        if response.status_code == 200:
            st.success("✅ Connected to Kaggle GPU service successfully!")
            return response.json()
        else:
            st.error(f"❌ GPU service error ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ **Timeout Error**: Kaggle service is taking too long. Please check if your Kaggle notebook is running.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 **Connection Error**: Cannot reach Kaggle GPU service. Please verify:")
        st.error("• Your Kaggle notebook is running")
        st.error("• ngrok tunnel is active")  
        st.error("• GPU_SERVICE_URL is correct")
        return None
    except Exception as e:
        st.error(f"❌ **Unexpected error**: {str(e)}")
        return None

def main():
    st.title("🔬 Blood Cell Analysis Platform")
    st.caption("Powered by SAM (Segment Anything Model) on Kaggle + Streamlit Cloud")
    
    # Service status indicator
    with st.sidebar:
        st.header("🔧 Service Configuration")
        
        # Check GPU service configuration
        gpu_url = None
        try:
            gpu_url = st.secrets.get("GPU_SERVICE_URL")
        except:
            gpu_url = os.getenv("GPU_SERVICE_URL")
        
        if gpu_url:
            st.success("✅ GPU Service Configured")
            if st.button("🔍 Test Connection"):
                try:
                    test_response = requests.get(f"{gpu_url}/health", timeout=10)
                    if test_response.status_code == 200:
                        st.success("🎉 Kaggle GPU service is online!")
                    else:
                        st.error("❌ GPU service not responding")
                except:
                    st.error("❌ Cannot connect to GPU service")
        else:
            st.error("❌ GPU Service Not Configured")
            with st.expander("📋 Setup Instructions"):
                st.markdown("""
                **To configure Kaggle GPU Service:**
                
                1. **Run your mock GPU service on Kaggle**
                2. **Expose with ngrok** in your Kaggle notebook
                3. **Copy the ngrok URL** (e.g., https://abc123.ngrok.io)
                4. **Add to Streamlit Cloud secrets:**
                   - Go to your app settings
                   - Add: `GPU_SERVICE_URL = "your-ngrok-url"`
                """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = BloodCellAnalyzer()
    
    # Analysis mode selection
    analysis_mode = st.sidebar.radio("Analysis Mode", 
                                   ["Single Analysis", "Compare Two Videos"])
    
    if analysis_mode == "Single Analysis":
        single_analysis_ui()
    else:
        comparison_ui()

def single_analysis_ui():
    """Single video analysis interface"""
    st.header("Upload Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov'],
        help="Upload blood cell video (max 200MB). Recommended: 5-second clips at 30fps"
    )
    
    if uploaded_file:
        video_bytes = uploaded_file.read()
        
        with st.spinner("Extracting frames..."):
            frames = extract_frames_from_video(video_bytes)
        
        if not frames:
            st.error("Could not extract frames from video.")
            return
        
        st.success(f"Extracted {len(frames)} frames")
        
        # Frame selection with thumbnails
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_idx = st.selectbox("Select frame for analysis", 
                                      range(len(frames)),
                                      format_func=lambda x: f"Frame {x + 1}")
            st.image(frames[selected_idx], caption=f"Selected Frame {selected_idx + 1}")
        
        with col2:
            st.subheader("Frame Thumbnails")
            # Show thumbnails
            for i, frame in enumerate(frames[:6]):  # Show max 6 thumbnails
                if st.button(f"Frame {i+1}", key=f"thumb_{i}", 
                           type="secondary" if i != selected_idx else "primary"):
                    st.rerun()
                st.image(frame, width=120, caption=f"Frame {i+1}")
        
        # Analysis section
        st.divider()
        
        col_settings, col_run = st.columns([1, 1])
        
        with col_settings:
            st.subheader("⚙️ Analysis Settings")
            use_circular_mask = st.checkbox("Use circular mask", value=True, 
                                          help="Apply circular mask to focus on blood sample area")
            k_neighbors = st.slider("K-neighbors for aggregation", 3, 10, 5,
                                  help="Number of nearest neighbors for aggregation analysis")
        
        with col_run:
            st.subheader("🚀 Run Analysis")
            if st.button("🔬 Analyze Blood Cells", type="primary", use_container_width=True):
                run_analysis(frames[selected_idx], use_circular_mask, k_neighbors)

def comparison_ui():
    """Two-video comparison interface with persistent results"""
    st.header("Compare Two Videos")
    
    # Progress indicator
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        video1_done = 'video1' in st.session_state.get('comparison_results', {})
        st.write("**Step 1:** Upload & Analyze Video 1")
        st.write("✅ Complete" if video1_done else "⏳ Pending")
    
    with progress_col2:
        video2_done = 'video2' in st.session_state.get('comparison_results', {})
        st.write("**Step 2:** Upload & Analyze Video 2") 
        st.write("✅ Complete" if video2_done else "⏳ Pending")
    
    with progress_col3:
        comparison_ready = video1_done and video2_done
        st.write("**Step 3:** View Comparison")
        st.write("🎉 Ready!" if comparison_ready else "⏳ Waiting")
    
    st.divider()
    
    # Initialize comparison results in session state
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    
    col1, col2 = st.columns(2)
    
    # Video 1
    with col1:
        st.subheader("📹 Video 1")
        video1 = st.file_uploader("Choose first video", 
                                type=['mp4', 'avi', 'mov'], 
                                key="video1")
        frames1 = None
        if video1:
            with st.spinner("Extracting frames from Video 1..."):
                frames1 = extract_frames_from_video(video1.read())
            
            frame1_idx = st.selectbox("Select frame", 
                                    range(len(frames1)) if frames1 else [],
                                    key="frame1",
                                    format_func=lambda x: f"Frame {x + 1}")
            if frames1:
                st.image(frames1[frame1_idx], caption="Video 1 - Selected Frame")
                
                if st.button("🔬 Analyze Video 1", key="analyze1", use_container_width=True):
                    with st.spinner("Analyzing Video 1..."):
                        try:
                            result1 = run_analysis_and_return_results(frames1[frame1_idx])
                            if result1:  # Only store if analysis succeeded
                                st.session_state.comparison_results['video1'] = result1
                                st.success("✅ Video 1 analysis complete!")
                        except Exception as e:
                            st.error(f"❌ Video 1 analysis failed: {str(e)}")
        
        # Show Video 1 status
        if 'video1' in st.session_state.comparison_results:
            st.info("✅ Video 1 analyzed and ready for comparison")
        else:
            st.info("⏳ Upload and analyze Video 1")
    
    # Video 2
    with col2:
        st.subheader("📹 Video 2")
        video2 = st.file_uploader("Choose second video", 
                                type=['mp4', 'avi', 'mov'], 
                                key="video2")
        frames2 = None
        if video2:
            with st.spinner("Extracting frames from Video 2..."):
                frames2 = extract_frames_from_video(video2.read())
            
            frame2_idx = st.selectbox("Select frame", 
                                    range(len(frames2)) if frames2 else [],
                                    key="frame2",
                                    format_func=lambda x: f"Frame {x + 1}")
            if frames2:
                st.image(frames2[frame2_idx], caption="Video 2 - Selected Frame")
                
                if st.button("🔬 Analyze Video 2", key="analyze2", use_container_width=True):
                    with st.spinner("Analyzing Video 2..."):
                        try:
                            result2 = run_analysis_and_return_results(frames2[frame2_idx])
                            if result2:  # Only store if analysis succeeded
                                st.session_state.comparison_results['video2'] = result2
                                st.success("✅ Video 2 analysis complete!")
                        except Exception as e:
                            st.error(f"❌ Video 2 analysis failed: {str(e)}")
        
        # Show Video 2 status
        if 'video2' in st.session_state.comparison_results:
            st.info("✅ Video 2 analyzed and ready for comparison")
        else:
            st.info("⏳ Upload and analyze Video 2")
    
    # Show comparison when both videos are analyzed
    if len(st.session_state.comparison_results) == 2:
        st.divider()
        
        # Clear comparison button
        col_clear, col_compare = st.columns([1, 2])
        with col_clear:
            if st.button("🗑️ Clear Results", help="Clear all analysis results"):
                st.session_state.comparison_results = {}
                st.rerun()
        
        with col_compare:
            st.success("🎉 Both videos analyzed! Comparison ready below.")
        
        # Display the comparison
        display_comparison(
            st.session_state.comparison_results['video1'], 
            st.session_state.comparison_results['video2']
        )
    
    elif len(st.session_state.comparison_results) == 1:
        st.info("📋 One video analyzed. Analyze the second video to see comparison.")
    
    else:
        st.info("📋 Upload and analyze both videos to see comparison.")

def run_analysis(frame, use_circular_mask=True, k_neighbors=5):
    """Run complete analysis and display results"""
    try:
        # Step 1: Get masks from Kaggle GPU service
        mask_response = call_mask_service(frame)
        
        if not mask_response or mask_response.get('status') != 'success':
            st.error("Analysis aborted due to GPU service failure.")
            return
        
        st.success(f"✅ Generated {len(mask_response['cell_masks'])} cell masks, "
                  f"{len(mask_response['cluster_masks'])} cluster masks")
        
        # Step 2: Run analysis locally (CPU)
        with st.spinner("🧮 Running blood cell analysis (CPU)..."):
            results = st.session_state.analyzer.analyze_from_masks(
                frame,
                mask_response['cell_masks'],
                mask_response['cluster_masks']
            )
        
        # Step 3: Display results
        display_results(results, frame)
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.exception(e)

def run_analysis_and_return_results(frame):
    """Run analysis and return results (for comparison)"""
    mask_response = call_mask_service(frame)
    if not mask_response or mask_response.get('status') != 'success':
        return None
    
    results = st.session_state.analyzer.analyze_from_masks(
        frame,
        mask_response['cell_masks'],
        mask_response['cluster_masks']
    )
    
    results['original_frame'] = frame
    return results

def display_results(results, original_image):
    """Display comprehensive analysis results"""
    st.subheader("📊 Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cells", results["total_cells"])
    with col2:
        st.metric("Individual Cells", results["individual_cells"])
    with col3:
        st.metric("Cluster Cells", results["estimated_cluster_cells"])
    with col4:
        mean_circ = np.mean(results['circularity_scores']) if results['circularity_scores'] else 0
        st.metric("Mean Circularity", f"{mean_circ:.3f}")
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        mean_agg = np.mean(results['aggregation_scores']) if len(results['aggregation_scores']) else 0
        st.metric("Mean Aggregation", f"{mean_agg:.1f}px")
    with col6:
        correction_factor = results['total_cells'] / results['individual_cells'] if results['individual_cells'] > 0 else 1
        st.metric("Correction Factor", f"{correction_factor:.2f}x")
    with col7:
        excellent_cells = sum(1 for score in results['circularity_scores'] if score >= 0.8) if len(results['circularity_scores']) else 0
        st.metric("Excellent Shape", f"{excellent_cells} cells")
    with col8:
        density_level = "HIGH" if mean_agg < 30 else "MEDIUM" if mean_agg < 50 else "LOW"
        st.metric("Density Level", density_level)
    
    # Visualizations
    st.subheader("🔍 Analysis Visualizations")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0,0].imshow(original_image)
    axes[0,0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0,0].axis('off')
    
    # Individual cells
    axes[0,1].imshow(original_image)
    cell_overlay = np.zeros((*original_image.shape[:2], 4))
    for mask in results['cell_masks']:
        cell_overlay[mask] = to_rgba('red', alpha=0.6)
    axes[0,1].imshow(cell_overlay)
    axes[0,1].set_title(f"Individual Cells ({len(results['cell_masks'])})", fontsize=14, fontweight='bold')
    axes[0,1].axis('off')
    
    # Clusters with estimated positions
    axes[0,2].imshow(original_image)
    cluster_overlay = np.zeros((*original_image.shape[:2], 4))
    for mask in results['cluster_masks']:
        cluster_overlay[mask] = to_rgba('green', alpha=0.5)
    axes[0,2].imshow(cluster_overlay)
    
    # Add estimated cell positions
    for info in results['estimated_cell_info']:
        for pos in info['positions']:
            axes[0,2].plot(pos[1], pos[0], 'yo', markersize=4, alpha=0.8)
    
    axes[0,2].set_title(f"Clusters ({len(results['cluster_masks'])}) + Estimated Cells", fontsize=14, fontweight='bold')
    axes[0,2].axis('off')
    
    # Circularity heatmap
    axes[1,0].imshow(original_image, cmap='gray')
    im_circ = axes[1,0].imshow(results['circularity_heatmap'], cmap='hot', alpha=0.7)
    axes[1,0].set_title("Circularity Heatmap", fontsize=14, fontweight='bold')
    axes[1,0].axis('off')
    plt.colorbar(im_circ, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    # Aggregation heatmap  
    axes[1,1].imshow(original_image, cmap='gray')
    im_agg = axes[1,1].imshow(results['unified_heatmap'], cmap='hot', alpha=0.7)
    
    # Mark all centroids
    for centroid in results['all_centroids'][:50]:  # Limit to 50 for visibility
        axes[1,1].plot(centroid[1], centroid[0], 'b+', markersize=3, alpha=0.7)
    
    axes[1,1].set_title("Unified Aggregation Heatmap", fontsize=14, fontweight='bold')
    axes[1,1].axis('off')
    plt.colorbar(im_agg, ax=axes[1,1], fraction=0.046, pad=0.04)
    
    # Combined view
    axes[1,2].imshow(original_image)
    combined = cell_overlay.copy()
    for mask in results['cluster_masks']:
        cluster_only = mask & ~(combined[:,:,0] > 0)
        combined[cluster_only] = to_rgba('green', alpha=0.5)
    axes[1,2].imshow(combined)
    axes[1,2].set_title("Combined View\n(Red: Individual, Green: Clusters)", fontsize=14, fontweight='bold')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed metrics
    with st.expander("📈 Detailed Analysis Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Circularity Distribution")
            if results['circularity_scores']:
                fig_hist, ax = plt.subplots(figsize=(8, 5))
                ax.hist(results['circularity_scores'], bins=20, alpha=0.7, color='blue')
                ax.set_xlabel('Circularity Score')
                ax.set_ylabel('Number of Cells')
                ax.set_title('Cell Circularity Distribution')
                st.pyplot(fig_hist)
        
        with col2:
            st.subheader("Aggregation Distribution")
            if len(results['aggregation_scores']):
                fig_hist2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.hist(results['aggregation_scores'], bins=20, alpha=0.7, color='green')
                ax2.set_xlabel('Aggregation Score (pixels)')
                ax2.set_ylabel('Number of Cells')
                ax2.set_title('Cell Aggregation Distribution')
                st.pyplot(fig_hist2)

def display_comparison(results1, results2):
    """Display side-by-side comparison of two analyses"""
    st.subheader("📊 Comparison Results")
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🎬 Video 1 Results**")
        st.metric("Total Cells", results1["total_cells"])
        st.metric("Individual Cells", results1["individual_cells"])
        st.metric("Cluster Cells", results1["estimated_cluster_cells"])
        mean_circ1 = np.mean(results1['circularity_scores']) if results1['circularity_scores'] else 0
        st.metric("Mean Circularity", f"{mean_circ1:.3f}")
        mean_agg1 = np.mean(results1['aggregation_scores']) if len(results1['aggregation_scores']) else 0
        st.metric("Mean Aggregation", f"{mean_agg1:.1f}px")
    
    with col2:
        st.write("**🎬 Video 2 Results**")
        st.metric("Total Cells", results2["total_cells"])
        st.metric("Individual Cells", results2["individual_cells"])
        st.metric("Cluster Cells", results2["estimated_cluster_cells"])
        mean_circ2 = np.mean(results2['circularity_scores']) if results2['circularity_scores'] else 0
        st.metric("Mean Circularity", f"{mean_circ2:.3f}")
        mean_agg2 = np.mean(results2['aggregation_scores']) if len(results2['aggregation_scores']) else 0
        st.metric("Mean Aggregation", f"{mean_agg2:.1f}px")
    
    # Difference analysis
    st.subheader("📈 Comparative Analysis")
    
    diff_col1, diff_col2, diff_col3, diff_col4 = st.columns(4)
    
    with diff_col1:
        total_diff = results2["total_cells"] - results1["total_cells"]
        st.metric("Total Cells Difference", 
                 total_diff, 
                 delta=total_diff)
    
    with diff_col2:
        individual_diff = results2["individual_cells"] - results1["individual_cells"]
        st.metric("Individual Cells Difference", 
                 individual_diff, 
                 delta=individual_diff)
    
    with diff_col3:
        circ_diff = mean_circ2 - mean_circ1
        st.metric("Circularity Difference", 
                 f"{circ_diff:.3f}", 
                 delta=f"{circ_diff:.3f}")
    
    with diff_col4:
        agg_diff = mean_agg2 - mean_agg1
        st.metric("Aggregation Difference", 
                 f"{agg_diff:.1f}px", 
                 delta=f"{agg_diff:.1f}px")
    
    # Side-by-side visualizations
    st.subheader("🔍 Side-by-Side Comparison")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Video 1 visualizations
    axes[0,0].imshow(results1['original_frame'])
    axes[0,0].set_title("Video 1: Original", fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(results1['original_frame'])
    cell_overlay1 = np.zeros((*results1['original_frame'].shape[:2], 4))
    for mask in results1['cell_masks']:
        cell_overlay1[mask] = to_rgba('red', alpha=0.6)
    axes[0,1].imshow(cell_overlay1)
    axes[0,1].set_title(f"Video 1: Cells ({len(results1['cell_masks'])})", fontweight='bold')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(results1['original_frame'], cmap='gray')
    axes[0,2].imshow(results1['circularity_heatmap'], cmap='hot', alpha=0.7)
    axes[0,2].set_title("Video 1: Circularity", fontweight='bold')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(results1['original_frame'], cmap='gray')
    axes[0,3].imshow(results1['unified_heatmap'], cmap='hot', alpha=0.7)
    axes[0,3].set_title("Video 1: Aggregation", fontweight='bold')
    axes[0,3].axis('off')
    
    # Video 2 visualizations
    axes[1,0].imshow(results2['original_frame'])
    axes[1,0].set_title("Video 2: Original", fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(results2['original_frame'])
    cell_overlay2 = np.zeros((*results2['original_frame'].shape[:2], 4))
    for mask in results2['cell_masks']:
        cell_overlay2[mask] = to_rgba('red', alpha=0.6)
    axes[1,1].imshow(cell_overlay2)
    axes[1,1].set_title(f"Video 2: Cells ({len(results2['cell_masks'])})", fontweight='bold')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(results2['original_frame'], cmap='gray')
    axes[1,2].imshow(results2['circularity_heatmap'], cmap='hot', alpha=0.7)
    axes[1,2].set_title("Video 2: Circularity", fontweight='bold')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(results2['original_frame'], cmap='gray')
    axes[1,3].imshow(results2['unified_heatmap'], cmp='hot', alpha=0.7)
    axes[1,3].set_title("Video 2: Aggregation", fontweight='bold')
    axes[1,3].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

main()