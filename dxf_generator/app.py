"""
DXF Canvas Composer API - Hugging Face Space Backend
Entry point: app.py (required by HF Spaces)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import json
import ezdxf
import requests
import uuid
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import shutil
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import numpy as np
import cv2
import numpy as np
from PIL import Image
import io
import base64
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale
from scipy.signal import savgol_filter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET_NAME = "lumashape"
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
HF_CACHE_DIR = os.getenv('HF_HOME', '/tmp/hf_cache')
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Data Classes
@dataclass
class ToolInfo:
    tool_id: str
    name: str
    brand: str
    dxf_link: str
    position_x_inches: float
    position_y_inches: float
    rotation_degrees: float = 0.0
    height_diagonal_inches: float = 0.0
    thickness_inches: float = 0.5
    flip_horizontal: bool = False
    flip_vertical: bool = False
    opacity: int = 100
    smooth: int = 0
    width_mm: float = 0.0
    height_mm: float = 0.0
    depth_mm: float = 0.0
    entities: List = None
    local_path: str = ""
    unit: str = ""
    is_custom_shape: bool = False
    shape_type: str = ""
    shape_data: Dict = None
    position_z_inches: float = 0.0
    cut_depth_inches: float = 0.0  # Converted to inches (for internal use)
    cut_type: str = "pocket"
    
    # NEW FIELDS for metadata display
    cut_depth_original_value: float = 0.0  # Store original value from payload
    cut_depth_original_unit: str = "inches"  # Store original unit

@dataclass
class CanvasConfig:
    width_inches: float
    height_inches: float
    thickness_inches: float
    has_overlaps: bool = False
    color: str = "natural"
    unit: str = "inches"
    original_width_value: float = 0.0
    original_height_value: float = 0.0
    original_thickness_value: float = 0.0

@dataclass
class LayoutMetadata:
    layout_name: str
    brand: str
    container_type: str

# S3 Manager
class S3Manager:
    def __init__(self):
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', AWS_REGION)
        
        if not self.aws_access_key or not self.aws_secret_key:
            logger.warning("AWS credentials not found")
            self.s3_client = None
        else:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                logger.info("S3 client initialized")
            except Exception as e:
                logger.error(f"S3 init failed: {e}")
                self.s3_client = None
    
    def upload_file(self, file_path: str, filename: str, content_type: str = None) -> Optional[str]:
        if not self.s3_client:
            return None
        
        try:
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            extra_args = {'ContentDisposition': 'inline'}
            
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_file(file_path, S3_BUCKET_NAME, unique_filename, ExtraArgs=extra_args)
            public_url = f"https://{S3_BUCKET_NAME}.s3.{self.aws_region}.amazonaws.com/{unique_filename}"
            logger.info(f"Uploaded to S3: {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return None
    
    def download_file(self, url: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            if 's3.amazonaws.com' in url or S3_BUCKET_NAME in url:
                parsed = urlparse(url)
                s3_key = parsed.path.lstrip('/')
                
                if self.s3_client:
                    self.s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
                    return True
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            logger.error(f"Download failed {url}: {e}")
            return False

# Tool Offset Configuration Manager
OFFSET_CONFIG_FILE = "tool_offset_config.json"

def get_tool_offset_inches():
    """Read tool offset from config file"""
    try:
        if os.path.exists(OFFSET_CONFIG_FILE):
            with open(OFFSET_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return float(config.get('tool_contour_offset_inches', 0.02))
    except Exception as e:
        logger.warning(f"Failed to read offset config: {e}")
    return 0.02  # Default fallback

def set_tool_offset_inches(offset_inches):
    """Write tool offset to config file"""
    try:
        config = {'tool_contour_offset_inches': float(offset_inches)}
        with open(OFFSET_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to write offset config: {e}")
        return False
        
# DXF Composer
class DXFCanvasComposer:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.cache_dir = os.path.join(HF_CACHE_DIR, f"dxf_cache_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def __del__(self):
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def _download_dxf_file(self, url: str, tool_id: str) -> Optional[str]:
        try:
            filename = f"{tool_id}_{uuid.uuid4().hex[:6]}.dxf"
            local_path = os.path.join(self.cache_dir, filename)
            
            if self.s3_manager.download_file(url, local_path):
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    try:
                        doc = ezdxf.readfile(local_path)
                        return local_path
                    except Exception as e:
                        logger.error(f"Invalid DXF: {e}")
                        os.remove(local_path)
            return None
        except Exception as e:
            logger.error(f"DXF download failed: {e}")
            return None
    
    def _extract_entity_points(self, entity):
        points = []
        try:
            entity_type = type(entity).__name__
            
            if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'start'):
                # LINE entity
                points = [entity.dxf.start, entity.dxf.end]
            elif entity_type == 'Spline' or entity_type == 'SPLINE':
                # SPLINE entity - this should NOT happen after conversion, but handle as fallback
                logger.warning(f"Encountered unconverted Spline entity - this shouldn't happen")
                try:
                    # Try fit_points
                    if hasattr(entity, 'fit_points') and entity.fit_points:
                        points = list(entity.fit_points)
                        logger.info(f"Extracted {len(points)} fit_points from Spline")
                        if len(points) > 0:
                            first_pt = points[0]
                            logger.info(f"Fit_points first point: type={type(first_pt).__name__}, has .x={hasattr(first_pt, 'x')}, is tuple/list={isinstance(first_pt, (tuple, list))}, repr={repr(first_pt)[:100]}")
                            if hasattr(first_pt, 'x'):
                                logger.info(f"First fit_point coords: x={first_pt.x}, y={first_pt.y}, z={getattr(first_pt, 'z', 'no z')}")
                    # Try control_points
                    elif hasattr(entity, 'control_points') and entity.control_points:
                        points = list(entity.control_points)
                        logger.info(f"Extracted {len(points)} control_points from Spline")
                    # Try flattening
                    else:
                        try:
                            flat_pts = list(entity.flattening(distance=0.1))
                            if flat_pts:
                                # DEBUG: Log the format of the first point
                                if len(flat_pts) > 0:
                                    first_pt = flat_pts[0]
                                    logger.info(f"Flattening returned {len(flat_pts)} points, first point type: {type(first_pt).__name__}, has .x: {hasattr(first_pt, 'x')}, is tuple/list: {isinstance(first_pt, (tuple, list))}")
                                    if hasattr(first_pt, 'x'):
                                        logger.info(f"First point coords: x={first_pt.x}, y={first_pt.y}")
                                    if hasattr(first_pt, '__dict__'):
                                        logger.info(f"First point dict: {first_pt.__dict__}")
                                points = flat_pts
                                logger.info(f"Extracted {len(points)} points via flattening")
                        except Exception as flatten_ex:
                            logger.warning(f"Flattening exception: {flatten_ex}")
                            pass
                except Exception as spline_ex:
                    logger.warning(f"Spline extraction failed: {spline_ex}")
                        
            elif hasattr(entity, 'vertices'):
                # POLYLINE/LWPOLYLINE entity
                try:
                    points = [vertex.dxf.location for vertex in entity.vertices()]
                except:
                    try:
                        vertices_data = list(entity.vertices())
                        points = [(v[0], v[1], 0) if len(v) >= 2 else (0, 0, 0) for v in vertices_data]
                    except:
                        points = []
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'center'):
                # CIRCLE/ARC entity
                center = entity.dxf.center
                radius = getattr(entity.dxf, 'radius', 0)
                points = [(center.x - radius, center.y - radius, center.z), (center.x + radius, center.y + radius, center.z)]
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'insert'):
                # INSERT entity
                points = [entity.dxf.insert]
        except Exception as e:
            logger.warning(f"Point extraction failed for {type(entity).__name__}: {e}")
        return points

    def _analyze_and_normalize_dxf(self, dxf_path: str) -> Tuple[float, float, float, List]:
        """
        Read a DXF file and return its bounding-box dimensions plus a list of
        raw numpy (N,2) point arrays — one per LWPolyline contour, already
        shifted so the bounding box starts at (0,0).

        Returning plain arrays (not ezdxf entity objects) means zero entity
        manipulation happens here, so the contour geometry is never corrupted.
        """
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            entities = list(msp)

            if not entities:
                logger.error(f"DXF file has no entities: {dxf_path}")
                return 0.0, 0.0, 0.0, []

            logger.info(f"DXF has {len(entities)} entities, types: {[type(e).__name__ for e in entities[:5]]}")

            # ── Collect raw (x,y) arrays from every LWPolyline ───────────────
            raw_segments = []   # list of np.ndarray shape (N,2)

            for entity in entities:
                etype = type(entity).__name__

                if etype == 'LWPolyline':
                    try:
                        pts = np.array(
                            [(x, y) for x, y in entity.get_points(format='xy')],
                            dtype=float
                        )
                        if len(pts) >= 2:
                            raw_segments.append(pts)
                    except Exception as ex:
                        logger.warning(f"LWPolyline read failed: {ex}")

                elif etype in ('Spline', 'SPLINE'):
                    # Flatten spline to points
                    try:
                        flat = list(entity.flattening(distance=0.1))
                        if flat:
                            pts = np.array(
                                [(p.x, p.y) if hasattr(p, 'x') else (p[0], p[1])
                                 for p in flat],
                                dtype=float
                            )
                            if len(pts) >= 2:
                                raw_segments.append(pts)
                    except Exception as ex:
                        logger.warning(f"Spline flatten failed: {ex}")

            if not raw_segments:
                logger.error(f"No usable contour points extracted from: {dxf_path}")
                return 0.0, 0.0, 0.0, []

            # ── Bounding box across all segments ─────────────────────────────
            all_pts = np.vstack(raw_segments)
            min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
            max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()

            width_mm  = float(max_x - min_x)
            height_mm = float(max_y - min_y)
            depth_mm  = 1.0   # DXF from image-to-dxf is 2D; placeholder

            logger.info(f"DXF dimensions: {width_mm:.2f}mm x {height_mm:.2f}mm  "
                        f"segments={len(raw_segments)}  total_pts={len(all_pts)}")

            # ── Shift every segment so bbox starts at (0,0) ──────────────────
            normalized_segments = [seg - [min_x, min_y] for seg in raw_segments]

            logger.info(f"Normalized {len(normalized_segments)} contour segment(s) to origin")
            return width_mm, height_mm, depth_mm, normalized_segments

        except Exception as e:
            logger.error(f"DXF analysis failed for {dxf_path}: {e}", exc_info=True)
            return 0.0, 0.0, 0.0, []

    def _scale_entities_to_target_size(self, segments, scale_factor: float):
        """
        Scale a list of numpy (N,2) point arrays by scale_factor.
        Replaces the old entity-object version.
        """
        if abs(scale_factor - 1.0) < 0.001:
            return segments
        return [seg * scale_factor for seg in segments]

    def _add_cutting_metadata(self, msp, canvas: CanvasConfig, tools: List[ToolInfo]):
        """Add text annotations with cutting depths and tool details"""
        try:
            if canvas.unit == "inches":
                canvas_width_mm = self._inches_to_mm(canvas.width_inches)
                canvas_height_mm = self._inches_to_mm(canvas.height_inches)
            else:
                canvas_width_mm = canvas.width_inches
                canvas_height_mm =canvas.height_inches
            
            # Position text outside canvas boundary (no buffer needed)
            text_x = canvas_width_mm + 10  # 10mm spacing
            text_y = canvas_height_mm - 10
            line_height = 5
            
            msp.add_text(
                "CUT DEPTHS",
                dxfattribs={
                    "layer": "METADATA",
                    "color": 7,
                    "height": 6
                }
            ).set_placement((text_x, text_y))
            
            text_y -= line_height * 2
            
            # Filter out FINGERCUT tools for numbering
            non_fingercut_tools = [t for t in tools if t.shape_type != "fingercut"]
            
            # Add tool-by-tool instructions with details
            tool_number = 1
            for i, tool in enumerate(tools):
                is_fingercut = tool.brand.upper() == "FINGERCUT"
                is_text = tool.shape_type == "text"
                
                # Skip text boxes in cutting instructions
                if is_text:
                    continue 
                
                # Get the cut_depth value (already stored in tool.cut_depth_inches)
                cut_depth_value = tool.cut_depth_inches
                
                # Build instruction text
                if is_fingercut:
                    label = "FINGERCUT"
                else:
                    label = f"Tool {tool_number}"
                    tool_number += 1
                
                # Format depth based on tool's unit
                unit = tool.cut_depth_original_unit
    
                if unit == "mm":
                    depth_str = f"{tool.cut_depth_original_value:.3f} mm"
                elif unit == "inches":
                    depth_str = f"{tool.cut_depth_original_value:.3f} in"
                else:
                    depth_str = f"{tool.cut_depth_original_value:.3f}"
                
                # ✅ BUILD THE INSTRUCTION STRING (this was missing!)
                # instruction = f"{label}: {tool.name}\nCut Depth: {depth_str}\nCut Type: {tool.cut_type}"
                instruction = f"{label}: {tool.name}\nCut Depth: {depth_str}\n"
                
                msp.add_mtext(
                    instruction,
                    dxfattribs={
                        "layer": "METADATA",
                        "color": 7,
                        "char_height": 3
                    }
                ).set_location((text_x, text_y))
                
                text_y -= line_height * 5
            
            # ===== CANVAS DETAILS SECTION ===== (moved outside loop)
            text_y -= line_height * 2
            
            # Add "CANVAS DETAILS" header
            msp.add_text(
                "CANVAS DETAILS",
                dxfattribs={
                    "layer": "METADATA",
                    "color": 6,
                    "height": 6
                }
            ).set_placement((text_x, text_y))
            
            text_y -= line_height * 2
            
            # Format values based on canvas unit
            canvas_unit = canvas.unit
            if canvas_unit == "mm":
                thickness_str = f"{canvas.original_thickness_value:.2f} mm"
                dimensions_str = f"{canvas.original_width_value:.2f} x {canvas.original_height_value:.2f} mm"
            else:  # inches (default)
                thickness_str = f"{canvas.thickness_inches:.3f} in"
                dimensions_str = f"{canvas.width_inches:.2f} x {canvas.height_inches:.2f} in"
            
            # Use canvas thickness as-is (already in inches)
            # thickness_str = f"{canvas.thickness_inches:.3f} in"
            # dimensions_str = f"{canvas.width_inches:.2f} x {canvas.height_inches:.2f} in"
            
            # Material thickness
            msp.add_text(
                f"Thickness: {thickness_str}",
                dxfattribs={
                    "layer": "METADATA",
                    "color": 7,
                    "height": 3.5
                }
            ).set_placement((text_x, text_y))
            
            text_y -= line_height * 1.2
            
            # Material color
            msp.add_text(
                f"Foam Color: {canvas.color.title()}",
                dxfattribs={
                    "layer": "METADATA",
                    "color": 7,
                    "height": 3.5
                }
            ).set_placement((text_x, text_y))
            
            text_y -= line_height * 1.2
            
            # Canvas dimensions
            msp.add_text(
                f"Dimensions: {dimensions_str}",
                dxfattribs={
                    "layer": "METADATA",
                    "color": 7,
                    "height": 3.5
                }
            ).set_placement((text_x, text_y))
            
        except Exception as e:
            logger.warning(f"Metadata annotation failed: {e}")
    
                            
    def _translate_entity_manually(self, entity, dx: float, dy: float, dz: float = 0):
        try:
            if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                entity.dxf.start = (entity.dxf.start.x + dx, entity.dxf.start.y + dy, entity.dxf.start.z + dz)
                entity.dxf.end = (entity.dxf.end.x + dx, entity.dxf.end.y + dy, entity.dxf.end.z + dz)
            elif hasattr(entity.dxf, 'center'):
                entity.dxf.center = (entity.dxf.center.x + dx, entity.dxf.center.y + dy, entity.dxf.center.z + dz)
            elif hasattr(entity.dxf, 'insert'):
                entity.dxf.insert = (entity.dxf.insert.x + dx, entity.dxf.insert.y + dy, entity.dxf.insert.z + dz)
            elif hasattr(entity, 'vertices'):
                entity_type = type(entity).__name__
                if entity_type == 'LWPolyline':
                    try:
                        current_vertices = list(entity.vertices())
                        new_vertices = []
                        for vertex in current_vertices:
                            if len(vertex) >= 2:
                                new_vertices.append((vertex[0] + dx, vertex[1] + dy))
                        entity.clear()
                        for vertex in new_vertices:
                            entity.append(vertex[:2])
                    except:
                        pass
        except:
            pass
    
    def _mm_to_inches(self, mm: float) -> float:
        return mm / 25.4
    
    def _inches_to_mm(self, inches: float) -> float:
        return inches * 25.4
    
  
    def _add_3d_canvas_representation(self, msp, canvas: CanvasConfig):
        try:
            # Use EXACT canvas dimensions - NO BUFFER
            # canvas_width_mm = self._inches_to_mm(canvas.width_inches)
            # canvas_height_mm = self._inches_to_mm(canvas.height_inches)
            # canvas_thickness_mm = self._inches_to_mm(canvas.thickness_inches)
            if canvas.unit == "inches":
                canvas_width_mm = self._inches_to_mm(canvas.width_inches)
                canvas_height_mm = self._inches_to_mm(canvas.height_inches)
                canvas_thickness_mm = self._inches_to_mm(canvas.thickness_inches)
            else:
                canvas_width_mm = canvas.width_inches
                canvas_height_mm = canvas.height_inches
                canvas_thickness_mm =canvas.thickness_inches
                
            # Create canvas boundary at EXACT dimensions
            msp.add_lwpolyline([
                (0, 0),
                (canvas_width_mm, 0),
                (canvas_width_mm, canvas_height_mm),
                (0, canvas_height_mm)
            ], close=True, dxfattribs={"layer": "CANVAS_BOTTOM", "color": 8})
            
            # Create top layer (3D representation)
            for entity in msp.query('LWPOLYLINE[layer=="CANVAS_BOTTOM"]'):
                top_entity = entity.copy()
                top_entity.dxf.layer = "CANVAS_TOP"
                top_entity.dxf.color = 6
                self._translate_entity_manually(top_entity, 0, 0, canvas_thickness_mm)
                msp.add_entity(top_entity)
            
            # Add vertical edges at corners
            corners = [
                (0, 0),
                (canvas_width_mm, 0),
                (canvas_width_mm, canvas_height_mm),
                (0, canvas_height_mm)
            ]
            for corner in corners:
                msp.add_line((corner[0], corner[1], 0), (corner[0], corner[1], canvas_thickness_mm),
                           dxfattribs={"layer": "CANVAS_EDGES", "color": 8})
                           
        except Exception as e:
            logger.warning(f"3D canvas failed: {e}")

    def _flip_entity_horizontal(self, entity, center_x: float):
        """Flip entity horizontally (mirror across vertical axis)"""
        try:
            if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                start_x = 2 * center_x - entity.dxf.start.x
                end_x = 2 * center_x - entity.dxf.end.x
                entity.dxf.start = (start_x, entity.dxf.start.y, entity.dxf.start.z)
                entity.dxf.end = (end_x, entity.dxf.end.y, entity.dxf.end.z)
            elif hasattr(entity.dxf, 'center'):
                cx = 2 * center_x - entity.dxf.center.x
                entity.dxf.center = (cx, entity.dxf.center.y, entity.dxf.center.z)
            elif hasattr(entity, 'vertices'):
                entity_type = type(entity).__name__
                if entity_type == 'LWPolyline':
                    current_vertices = list(entity.vertices())
                    new_vertices = []
                    for vertex in current_vertices:
                        if len(vertex) >= 2:
                            new_x = 2 * center_x - vertex[0]
                            new_vertices.append((new_x, vertex[1]))
                    entity.clear()
                    for vertex in new_vertices:
                        entity.append(vertex[:2])
        except Exception as e:
            logger.warning(f"Horizontal flip failed: {e}")

    def _flip_entity_vertical(self, entity, center_y: float):
        """Flip entity vertically (mirror across horizontal axis)"""
        try:
            if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                start_y = 2 * center_y - entity.dxf.start.y
                end_y = 2 * center_y - entity.dxf.end.y
                entity.dxf.start = (entity.dxf.start.x, start_y, entity.dxf.start.z)
                entity.dxf.end = (entity.dxf.end.x, end_y, entity.dxf.end.z)
            elif hasattr(entity.dxf, 'center'):
                cy = 2 * center_y - entity.dxf.center.y
                entity.dxf.center = (entity.dxf.center.x, cy, entity.dxf.center.z)
            elif hasattr(entity, 'vertices'):
                entity_type = type(entity).__name__
                if entity_type == 'LWPolyline':
                    current_vertices = list(entity.vertices())
                    new_vertices = []
                    for vertex in current_vertices:
                        if len(vertex) >= 2:
                            new_y = 2 * center_y - vertex[1]
                            new_vertices.append((vertex[0], new_y))
                    entity.clear()
                    for vertex in new_vertices:
                        entity.append(vertex[:2])
        except Exception as e:
            logger.warning(f"Vertical flip failed: {e}")

        
    def _place_lwpolyline_tool(self, msp, tool, layer_name: str,
                               offset_x_mm: float, offset_y_mm: float,
                               offset_z_mm: float, cut_depth_mm: float) -> int:
        """
        Place a DXF tool contour on the canvas.
        tool.entities is now a list of numpy (N,2) arrays (pure point data,
        no ezdxf entity objects), so zero entity manipulation happens here.
        Flip/rotate/translate are plain numpy ops, then one fresh LWPolyline
        is written — guaranteed identical geometry to image-to-dxf output.
        """
        import math as _math

        segments = tool.entities   # list of np.ndarray (N,2)
        if not segments:
            logger.warning(f"Tool '{tool.name}': no contour segments — skipping")
            return 0

        cx = tool.width_mm  / 2.0
        cy = tool.height_mm / 2.0
        entities_written = 0

        for pts in segments:
            pts = pts.copy().astype(float)   # never mutate the stored array

            # ── Flip ─────────────────────────────────────────────────────────
            if tool.flip_horizontal:
                pts[:, 0] = 2.0 * cx - pts[:, 0]
            if tool.flip_vertical:
                pts[:, 1] = 2.0 * cy - pts[:, 1]

            # ── Rotate around tool centre ─────────────────────────────────────
            if tool.rotation_degrees != 0:
                angle_rad = _math.radians(-tool.rotation_degrees)
                cos_a, sin_a = _math.cos(angle_rad), _math.sin(angle_rad)
                pts -= [cx, cy]
                pts  = np.column_stack([
                    pts[:, 0] * cos_a - pts[:, 1] * sin_a,
                    pts[:, 0] * sin_a + pts[:, 1] * cos_a,
                ])
                pts += [cx, cy]

            # ── Translate to canvas position ──────────────────────────────────
            pts += [offset_x_mm, offset_y_mm]

            # ── Write fresh LWPolyline ────────────────────────────────────────
            try:
                msp.add_lwpolyline(
                    pts.tolist(),
                    close=True,
                    dxfattribs={
                        "layer":     layer_name,
                        "color":     1,
                        "elevation": offset_z_mm,
                        "thickness": cut_depth_mm,
                    }
                )
                entities_written += 1
            except Exception as ex:
                logger.warning(f"Tool '{tool.name}': LWPolyline write failed — {ex}")

        logger.info(f"Tool '{tool.name}': placed {entities_written} contour(s) via pure-numpy transform")
        return entities_written

    def _create_shape_entities(self, shape_type: str, shape_data: Dict) -> Tuple[float, float, float, List]:
        """
        Create DXF entities from shape parameters
        Returns: (width_mm, height_mm, depth_mm, entities)
        """
        entities = []
        
        try:
            if shape_type == "rectangle":
                width_inches = shape_data.get("width_inches", 1.0)
                height_inches = shape_data.get("height_inches", 1.0)
                width_mm = self._inches_to_mm(width_inches)
                height_mm = self._inches_to_mm(height_inches)
                
                # Create rectangle polyline (normalized to origin)
                points = [
                    (0, 0),
                    (width_mm, 0),
                    (width_mm, height_mm),
                    (0, height_mm)
                ]
                # Create a temporary doc to generate the entity
                temp_doc = ezdxf.new()
                temp_msp = temp_doc.modelspace()
                lwpolyline = temp_msp.add_lwpolyline(points, close=True)
                entities.append(lwpolyline)
                
                return width_mm, height_mm, 1.0, entities
            
            elif shape_type == "circle":
                radius_inches = shape_data.get("radius_inches", 1.0)
                radius_mm = self._inches_to_mm(radius_inches)
                diameter_mm = radius_mm * 2
                
                # Create circle entity (normalized to origin)
                # Center at (radius, radius) so bounding box starts at (0, 0)
                temp_doc = ezdxf.new()
                temp_msp = temp_doc.modelspace()
                circle = temp_msp.add_circle(center=(radius_mm, radius_mm), radius=radius_mm)
                entities.append(circle)
                
                return diameter_mm, diameter_mm, 1.0, entities
            
            elif shape_type == "fingercut":
                width_inches = shape_data.get("width_inches", 1.0)
                height_inches = shape_data.get("height_inches", 1.0)
                width_mm = self._inches_to_mm(width_inches)
                height_mm = self._inches_to_mm(height_inches)
            
                temp_doc = ezdxf.new()
                temp_msp = temp_doc.modelspace()
            
                # Radius = half of the *shorter* side → bulging semicircles on shorter ends only
                radius = min(width_mm, height_mm) / 2.0
                segments = 30  # Smooth curve
            
                points = []
            
                if width_mm < height_mm:
                    # VERTICAL TUBE: width is shorter
                    # Semicircles bulge outward on TOP and BOTTOM ends
                    # Left and right are STRAIGHT lines
                    
                    straight_height = height_mm - 2 * radius
                    
                    # Start at bottom-left corner
                    points.append((0, radius))
                    
                    # LEFT STRAIGHT LINE (upward)
                    points.append((0, radius + straight_height))
                    
                    # TOP SEMICIRCLE (left to right, bulging upward)
                    for i in range(segments + 1):
                        angle = np.pi + (np.pi * i / segments)  # π → 2π
                        x = radius + radius * np.cos(angle)
                        y = height_mm - radius + radius * np.sin(angle)
                        points.append((x, y))
                    
                    # RIGHT STRAIGHT LINE (downward)
                    points.append((width_mm, radius + straight_height))
                    points.append((width_mm, radius))
                    
                    # BOTTOM SEMICIRCLE (right to left, bulging downward)
                    for i in range(segments + 1):
                        angle = 0 - (np.pi * i / segments)  # 0 → -π
                        x = radius + radius * np.cos(angle)
                        y = radius + radius * np.sin(angle)
                        points.append((x, y))
                    
                else:
                    # HORIZONTAL TUBE: height is shorter
                    # Semicircles bulge outward on LEFT and RIGHT ends
                    # Top and bottom are STRAIGHT lines
                    
                    straight_width = width_mm - 2 * radius
                    
                    # Start at bottom-left corner
                    points.append((radius, 0))
                    
                    # BOTTOM STRAIGHT LINE (rightward)
                    points.append((radius + straight_width, 0))
                    
                    # RIGHT SEMICIRCLE (bottom to top, bulging rightward)
                    for i in range(segments + 1):
                        angle = -np.pi/2 + (np.pi * i / segments)  # -π/2 → π/2
                        x = width_mm - radius + radius * np.cos(angle)
                        y = radius + radius * np.sin(angle)
                        points.append((x, y))
                    
                    # TOP STRAIGHT LINE (leftward)
                    points.append((radius + straight_width, height_mm))
                    points.append((radius, height_mm))
                    
                    # LEFT SEMICIRCLE (top to bottom, bulging leftward)
                    for i in range(segments + 1):
                        angle = np.pi/2 + (np.pi * i / segments)  # π/2 → 3π/2
                        x = radius + radius * np.cos(angle)
                        y = radius + radius * np.sin(angle)
                        points.append((x, y))
            
                # Create closed polyline
                lwpolyline = temp_msp.add_lwpolyline(points, close=True)
                entities.append(lwpolyline)
            
                return width_mm, height_mm, 1.0, entities
                
            elif shape_type == "text":
                # Extract text properties
                width_inches = shape_data.get("width_inches", 1.0)
                height_inches = shape_data.get("height_inches", 0.5)
                text_content = shape_data.get("content", "")
                font_size_px = shape_data.get("font_size_px", 12)
                text_align = shape_data.get("align", "center")  # "left", "center", "right"
                text_color = shape_data.get("color", "#000000")
                
                width_mm = self._inches_to_mm(width_inches)
                height_mm = self._inches_to_mm(height_inches)
                
                temp_doc = ezdxf.new()
                temp_msp = temp_doc.modelspace()
                
                # Create rectangle boundary for text box
                points = [
                    (0, 0),
                    (width_mm, 0),
                    (width_mm, height_mm),
                    (0, height_mm)
                ]
                # rectangle = temp_msp.add_lwpolyline(points, close=True)
                # entities.append(rectangle)
                
                # Calculate text position based on alignment
                padding_mm = 1.5
                if text_align == "left":
                    text_x = padding_mm
                    text_alignment = ezdxf.enums.TextEntityAlignment.LEFT
                elif text_align == "right":
                    text_x = width_mm - padding_mm
                    text_alignment = ezdxf.enums.TextEntityAlignment.RIGHT
                else:  # center (default)
                    text_x = width_mm / 2
                    text_alignment = ezdxf.enums.TextEntityAlignment.CENTER
                
                # text_y = height_mm / 2 
                text_y = (height_mm / 2) - (height_mm * 0.15)
                
                # Calculate dynamic text height based on box dimensions
                # Use 55% of the box height (reduced from 70%)
                # Use 55% of the box height
                text_height_mm = height_mm * 0.70
                
                # Also consider width constraints for long text
                # Estimate character width (roughly 0.6 * height for typical fonts)
                estimated_char_width = text_height_mm * 0.6
                estimated_text_width = len(text_content) * estimated_char_width
                
                # If text is too wide, reduce height proportionally
                if estimated_text_width > (width_mm * 0.88):  # Leave 12% padding
                    text_height_mm = (width_mm * 0.88) / (len(text_content) * 0.6)
                
                # Cap maximum text height to prevent oversized text
                max_text_height_mm = 7.0
                text_height_mm = min(text_height_mm, max_text_height_mm)
                
                # Ensure minimum text height for readability
                min_text_height_mm = 1.5
                text_height_mm = max(text_height_mm, min_text_height_mm)
                
                # Parse hex color to DXF color index (simplified - use color 7 for custom colors)
                dxf_color = 7  # White/Black depending on background
                
                # Add text entity
                text_entity = temp_msp.add_text(
                    text_content,
                    dxfattribs={
                        "layer": "TEXT_CONTENT",
                        "color": dxf_color,
                        "height": text_height_mm
                    }
                )
                text_entity.set_placement((text_x, text_y), align=text_alignment)
                entities.append(text_entity)
                
                return width_mm, height_mm, 0.1, entities  # Minimal depth for text
                
            else:
                raise ValueError(f"Unsupported shape type: {shape_type}. Supported types are: rectangle, circle, fingercut, text")
                
        except Exception as e:
            logger.error(f"Shape creation failed: {e}")
            return 0.0, 0.0, 0.0, []
        
    def parse_layout_json(self, layout_data: Dict) -> Tuple[CanvasConfig, LayoutMetadata, List[ToolInfo]]:
        canvas_info = layout_data.get("canvas_information", {})
        
        # Extract unit from canvas_information
        canvas_unit = canvas_info.get("unit", "inches").lower()
        
        # Get raw values
        width_value = canvas_info.get("width_inches", 21.0)
        height_value = canvas_info.get("height_inches", 11.0)
        thickness_value = canvas_info.get("thickness_inches", 0.5)
        
        # Convert to inches if needed (internal representation is always inches)
        # if canvas_unit == "mm":
        #     width_inches = width_value / 25.4
        #     height_inches = height_value / 25.4
        #     thickness_inches = thickness_value / 25.4
        # else:
        width_inches = width_value
        height_inches = height_value
        thickness_inches = thickness_value
        
        canvas = CanvasConfig(
            width_inches=width_inches,
            height_inches=height_inches,
            thickness_inches=thickness_inches,
            has_overlaps=canvas_info.get("has_overlaps", False),
            color=canvas_info.get("canvas_color", "natural"),
            unit=canvas_unit  # ← STORE ORIGINAL UNIT
        )
        canvas.original_width_value = width_value
        canvas.original_height_value = height_value
        canvas.original_thickness_value = thickness_value
        
        layout_info = layout_data.get("layout_metadata", {})
        metadata = LayoutMetadata(
            layout_name=layout_info.get("layout_name", "Unknown"),
            brand=layout_info.get("brand", "Unknown"),
            container_type=layout_info.get("container_type", "Drawer")
        )
        
        tools = []
        for tool_data in layout_data.get("tools", []):
            is_custom = tool_data.get("is_custom_shape", False)
            
            # Get unit and convert if needed
            unit = tool_data.get("unit", "inches").lower()
            
            # Get position data (with z support)
            position_data = tool_data.get("position_inches", {})
            thickness = tool_data.get("thickness_inches", 0.5)
            
            # Get depth values (ORIGINAL VALUES from payload)
            depth_inches_value = tool_data.get("depth_inches", 0)
            cut_depth_value = tool_data.get("cut_depth_inches", thickness)
            height_diagonal_value = tool_data.get("height_diagonal_inches", 0.0)
            
            # Store original cut_depth for metadata display
            cut_depth_original_value = cut_depth_value
            cut_depth_original_unit = unit
            
            # Convert measurements to inches for internal calculations
            # if unit == 'mm':
            #     # Convert mm to inches for internal use
            #     height_diagonal_inches = height_diagonal_value / 25.4
            #     depth_inches = depth_inches_value / 25.4
            #     cut_depth_inches = cut_depth_value / 25.4
            #     logger.info(f"Tool '{tool_data.get('name', 'Unknown')}': Converted from mm - "
            #                f"height: {height_diagonal_value}mm -> {height_diagonal_inches:.3f}in, "
            #                f"depth: {depth_inches_value}mm -> {depth_inches:.3f}in, "
            #                f"cut_depth: {cut_depth_value}mm -> {cut_depth_inches:.3f}in")


            # elif unit == 'inches':
            #     # Use values as-is
            #     height_diagonal_inches = height_diagonal_value
            #     depth_inches = depth_inches_value
            #     cut_depth_inches = cut_depth_value
            # else:
            height_diagonal_inches = height_diagonal_value
            depth_inches = depth_inches_value
            cut_depth_inches = cut_depth_value
            # logger.warning(f"Invalid unit '{unit}' for tool {tool_data.get('tool_id', 'unknown')}, defaulting to inches")
            # height_diagonal_inches = height_diagonal_value
            # depth_inches = depth_inches_value
            # cut_depth_inches = cut_depth_value
            
            tool = ToolInfo(
                tool_id=tool_data.get("tool_id", ""),
                name=tool_data.get("name", "Unknown"),
                brand=tool_data.get("brand", "Unknown"),
                dxf_link=tool_data.get("dxf_link", ""),
                position_x_inches=position_data.get("x", 0.0),
                position_y_inches=position_data.get("y", 0.0),
                position_z_inches=position_data.get("z", 0.0),
                rotation_degrees=tool_data.get("rotation_degrees", 0.0),
                height_diagonal_inches=height_diagonal_inches,
                thickness_inches=thickness,
                flip_horizontal=tool_data.get("flip_horizontal", False),
                flip_vertical=tool_data.get("flip_vertical", False),
                opacity=tool_data.get("opacity", 100),
                smooth=tool_data.get("smooth", 0),
                unit=unit,
                cut_depth_inches=cut_depth_inches,  # Converted for internal use
                cut_type=tool_data.get("cut_type", "pocket"),
                # Store original values for metadata display
                cut_depth_original_value=cut_depth_original_value,
                cut_depth_original_unit=cut_depth_original_unit,
                # Existing custom shape fields
                is_custom_shape=is_custom,
                shape_type=tool_data.get("shape_type", ""),
                shape_data=tool_data.get("shape_data", {})
            )
            tools.append(tool)
        
        return canvas, metadata, tools

    
    def _get_outer_contour_only(self, entities: List) -> List:
        """
        Extract only the outer contour from entities.
        Filters out internal details, keeping only the main boundary.
        """
        try:
            # Collect all polylines and lines
            closed_contours = []
            open_polylines = []
            lines = []
            other_entities = []
            
            for entity in entities:
                entity_type = type(entity).__name__
                
                if entity_type == 'LWPolyline':
                    if entity.closed or entity.is_closed:
                        # Calculate perimeter for closed polylines
                        try:
                            points = list(entity.vertices())
                            if len(points) >= 3:
                                perimeter = 0
                                for i in range(len(points)):
                                    p1 = points[i]
                                    p2 = points[(i + 1) % len(points)]
                                    dx = p2[0] - p1[0]
                                    dy = p2[1] - p1[1]
                                    perimeter += (dx*dx + dy*dy) ** 0.5
                                closed_contours.append((perimeter, entity))
                        except:
                            closed_contours.append((0, entity))
                    else:
                        # Keep open polylines as fallback
                        open_polylines.append(entity)
                        
                elif entity_type == 'Polyline':
                    if entity.is_closed:
                        try:
                            points = [v.dxf.location for v in entity.vertices()]
                            if len(points) >= 3:
                                perimeter = 0
                                for i in range(len(points)):
                                    p1 = points[i]
                                    p2 = points[(i + 1) % len(points)]
                                    dx = p2[0] - p1[0]
                                    dy = p2[1] - p1[1]
                                    perimeter += (dx*dx + dy*dy) ** 0.5
                                closed_contours.append((perimeter, entity))
                        except:
                            closed_contours.append((0, entity))
                    else:
                        open_polylines.append(entity)
                        
                elif entity_type == 'Line':
                    lines.append(entity)
                else:
                    other_entities.append(entity)
            
            # Priority 1: Return largest closed contour if found
            if closed_contours:
                closed_contours.sort(reverse=True, key=lambda x: x[0])
                return [closed_contours[0][1]]
            
            # Priority 2: If no closed contours, return all open polylines + lines
            if open_polylines or lines:
                logger.warning("No closed contours found, using open polylines and lines")
                return open_polylines + lines + other_entities
            
            # Priority 3: Return all entities as last resort
            logger.warning("No recognizable contours, returning all entities")
            return entities
            
        except Exception as e:
            logger.warning(f"Contour filtering failed: {e}")
            return entities  # Safe fallback
            

    def _add_tool_labels(self, msp, tools: List[ToolInfo]):
        """Add text labels under each tool on the canvas"""
        try:
            for i, tool in enumerate(tools):
                # Calculate tool center position
                offset_x_mm = self._inches_to_mm(tool.position_x_inches)
                offset_y_mm = self._inches_to_mm(tool.position_y_inches)
                
                # Position text at bottom-center of tool
                text_x = offset_x_mm + (tool.width_mm / 2)
                text_y = offset_y_mm - 3  # 3mm below tool bottom edge
                
                # Create label text
                label = f"Tool {i+1}"
                
                # Add text entity
                msp.add_text(
                    label,
                    dxfattribs={
                        "layer": "TOOL_LABELS",
                        "color": 3,  # Green color
                        "height": 2.5  # Text height in mm
                    }
                ).set_placement(
                    (text_x, text_y),
                    align=ezdxf.enums.TextEntityAlignment.BOTTOM_CENTER
                )
                
                # Optionally add tool name below label
                if tool.name and tool.name != "Unknown":
                    name_y = text_y - 3  # 3mm below label
                    msp.add_text(
                        tool.name[:20],  # Truncate long names
                        dxfattribs={
                            "layer": "TOOL_LABELS",
                            "color": 3,
                            "height": 1.8
                        }
                    ).set_placement(
                        (text_x, name_y),
                        align=ezdxf.enums.TextEntityAlignment.BOTTOM_CENTER
                    )
                    
        except Exception as e:
            logger.warning(f"Tool label annotation failed: {e}")
    
    # def _add_tool_numbers_inside(self, msp, tools: List[ToolInfo]):
    #     """Add small numbers inside each tool/shape on the canvas with smart placement around finger cuts"""
    #     try:
    #         tool_number = 1
    #         for i, tool in enumerate(tools):
    #             if tool.shape_type == "text":  # ADD THIS
    #                 continue  # ADD THIS

                
    #             # Calculate tool center position
    #             offset_x_mm = self._inches_to_mm(tool.position_x_inches)
    #             offset_y_mm = self._inches_to_mm(tool.position_y_inches)
                
    #             # Default position: center of tool
    #             text_x = offset_x_mm + (tool.width_mm / 2)
    #             text_y = offset_y_mm + (tool.height_mm / 2)
                
    #             # For FINGERCUT: always use exact center (no adjustment)
    #             if tool.shape_type == "fingercut":
    #                 # Place number at exact center of fingercut
    #                 msp.add_text(
    #                     str(tool_number),
    #                     dxfattribs={
    #                         "layer": "TOOL_NUMBERS",
    #                         "color": 5,  # Blue color
    #                         "height": 3.0  # Small text height in mm
    #                     }
    #                 ).set_placement(
    #                     (text_x, text_y),
    #                     align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER
    #                 )
    #                 tool_number += 1
    #                 continue
                
    #             # For OTHER SHAPES: check if fingercuts overlap and adjust position
    #             overlapping_finger_cuts = self._find_overlapping_finger_cuts(tool, tools)
                
    #             if overlapping_finger_cuts:
    #                 # Smart placement to avoid finger cuts
    #                 text_x, text_y = self._calculate_smart_number_position(
    #                     tool, overlapping_finger_cuts, offset_x_mm, offset_y_mm
    #                 )
                
    #             # Add small text entity at calculated position
    #             msp.add_text(
    #                 str(tool_number),
    #                 dxfattribs={
    #                     "layer": "TOOL_NUMBERS",
    #                     "color": 5,  # Blue color
    #                     "height": 3.0  # Small text height in mm
    #                 }
    #             ).set_placement(
    #                 (text_x, text_y),
    #                 align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER
    #             )
                
    #             tool_number += 1
                        
    #     except Exception as e:
    #         logger.warning(f"Tool number annotation failed: {e}")
    def _add_tool_numbers_inside(self, msp, tools: List[ToolInfo]):
        """Add small numbers inside each tool/shape on the canvas"""
        try:
            tool_number = 1
            for i, tool in enumerate(tools):
                # Skip ONLY text boxes and fingercuts
                if tool.shape_type == "text":
                    continue
                # if tool.shape_type == "fingercut":
                #     continue
    
                # Calculate the FINAL transformed position by finding the actual entity bounds
                # Get entities for this tool from the modelspace
                tool_layer = None
                for entity in msp:
                    if hasattr(entity.dxf, 'layer'):
                        layer_name = entity.dxf.layer
                        if (layer_name.startswith(f"TOOL_{i+1}_") or 
                            layer_name.startswith(f"CUSTOM_{tool.shape_type.upper()}_{i+1}")):
                            tool_layer = layer_name
                            break
                            
                
                if not tool_layer:
                    tool_number += 1
                    continue
                
                # Find all entities on this layer and calculate their actual bounding box
                # min_x = min_y = float('inf')
                # max_x = max_y = float('-inf')
                
                # for entity in msp.query(f'*[layer=="{tool_layer}"]'):
                #     points = self._extract_entity_points(entity)
                #     for point in points:
                #         if hasattr(point, 'x'):
                #             x, y = point.x, point.y
                #         elif isinstance(point, (tuple, list)) and len(point) >= 2:
                #             x, y = point[0], point[1]
                #         else:
                #             continue
                #         min_x = min(min_x, x)
                #         max_x = max(max_x, x)
                #         min_y = min(min_y, y)
                #         max_y = max(max_y, y)

                min_x = min_y = float('inf')
                max_x = max_y = float('-inf')
                
                for entity in msp.query(f'*[layer=="{tool_layer}"]'):
                    # Special handling for CIRCLE entities
                    entity_type = type(entity).__name__
                    if entity_type == 'Circle':
                        center = entity.dxf.center
                        radius = entity.dxf.radius
                        min_x = min(min_x, center.x - radius)
                        max_x = max(max_x, center.x + radius)
                        min_y = min(min_y, center.y - radius)
                        max_y = max(max_y, center.y + radius)
                        continue
                    
                    # For other entity types, use point extraction
                    points = self._extract_entity_points(entity)
                    for point in points:
                        if hasattr(point, 'x'):
                            x, y = point.x, point.y
                        elif isinstance(point, (tuple, list)) and len(point) >= 2:
                            x, y = point[0], point[1]
                        else:
                            continue
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
                        
                
                if min_x == float('inf'):
                    tool_number += 1
                    continue
                
                # Calculate center of the TRANSFORMED bounding box
                text_x = (min_x + max_x) / 2
                text_y = (min_y + max_y) / 2
                
                # For FINGERCUT: use exact center (already calculated above)
                # For OTHER SHAPES: check if fingercuts overlap and adjust position
                if tool.shape_type != "fingercut":
                    overlapping_finger_cuts = self._find_overlapping_finger_cuts(tool, tools)
                    
                    if overlapping_finger_cuts:
                        # Adjust position to avoid finger cuts
                        # Use a corner position instead of center
                        margin_x = (max_x - min_x) * 0.2
                        margin_y = (max_y - min_y) * 0.2
                        
                        # Try top-right corner
                        text_x = max_x - margin_x
                        text_y = max_y - margin_y
                
                # Add text entity at calculated position
                msp.add_text(
                    str(tool_number),
                    dxfattribs={
                        "layer": "TOOL_NUMBERS",
                        "color": 5,  # Blue color
                        "height": 3.0  # Small text height in mm
                    }
                ).set_placement(
                    (text_x, text_y),
                    align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER
                )
                
                tool_number += 1
                        
        except Exception as e:
            logger.warning(f"Tool number annotation failed: {e}")
        
    
    def _find_overlapping_finger_cuts(self, tool: ToolInfo, all_tools: List[ToolInfo]) -> List[ToolInfo]:
        """Find finger cuts that overlap with the given tool"""
        overlapping_cuts = []
        
        try:
            # Get tool bounding box
            tool_left = tool.position_x_inches
            tool_right = tool.position_x_inches + self._mm_to_inches(tool.width_mm)
            tool_bottom = tool.position_y_inches
            tool_top = tool.position_y_inches + self._mm_to_inches(tool.height_mm)
            
            for other_tool in all_tools:
                # Only check finger cuts
                if other_tool.shape_type != "fingercut":
                    continue
                    
                # Get finger cut bounding box
                cut_left = other_tool.position_x_inches
                cut_right = other_tool.position_x_inches + self._mm_to_inches(other_tool.width_mm)
                cut_bottom = other_tool.position_y_inches
                cut_top = other_tool.position_y_inches + self._mm_to_inches(other_tool.height_mm)
                
                # Check for overlap
                if (tool_left < cut_right and tool_right > cut_left and 
                    tool_bottom < cut_top and tool_top > cut_bottom):
                    overlapping_cuts.append(other_tool)
                    
        except Exception as e:
            logger.warning(f"Overlap detection failed: {e}")
        
        return overlapping_cuts
    
    def _calculate_smart_number_position(self, tool: ToolInfo, finger_cuts: List[ToolInfo], 
                                       offset_x_mm: float, offset_y_mm: float) -> Tuple[float, float]:
        """Calculate optimal number position to avoid finger cuts"""
        try:
            tool_width_mm = tool.width_mm
            tool_height_mm = tool.height_mm
            
            # Calculate safe margins (10% from edges)
            margin_x = tool_width_mm * 0.1
            margin_y = tool_height_mm * 0.1
            
            # Get the primary finger cut position relative to tool
            if finger_cuts:
                primary_cut = finger_cuts[0]
                cut_center_x = self._inches_to_mm(primary_cut.position_x_inches) + (primary_cut.width_mm / 2)
                cut_center_y = self._inches_to_mm(primary_cut.position_y_inches) + (primary_cut.height_mm / 2)
                
                # Convert to tool-relative coordinates
                cut_rel_x = cut_center_x - offset_x_mm
                cut_rel_y = cut_center_y - offset_y_mm
                
                # Determine which quadrant the finger cut is in relative to tool center
                tool_center_x = tool_width_mm / 2
                tool_center_y = tool_height_mm / 2
                
                # Calculate available quadrants (avoiding the finger cut area)
                quadrants = []
                
                # Top-left quadrant
                if cut_rel_x > tool_center_x or cut_rel_y > tool_center_y:
                    quadrants.append(("top-left", margin_x, tool_height_mm - margin_y))
                
                # Top-right quadrant  
                if cut_rel_x < tool_center_x or cut_rel_y > tool_center_y:
                    quadrants.append(("top-right", tool_width_mm - margin_x, tool_height_mm - margin_y))
                
                # Bottom-left quadrant
                if cut_rel_x > tool_center_x or cut_rel_y < tool_center_y:
                    quadrants.append(("bottom-left", margin_x, margin_y))
                
                # Bottom-right quadrant
                if cut_rel_x < tool_center_x or cut_rel_y < tool_center_y:
                    quadrants.append(("bottom-right", tool_width_mm - margin_x, margin_y))
                
                # Choose the quadrant farthest from the finger cut
                if quadrants:
                    best_quadrant = None
                    max_distance = -1
                    
                    for quadrant_name, qx, qy in quadrants:
                        distance = ((qx - cut_rel_x) ** 2 + (qy - cut_rel_y) ** 2) ** 0.5
                        if distance > max_distance:
                            max_distance = distance
                            best_quadrant = (qx, qy)
                    
                    if best_quadrant:
                        return offset_x_mm + best_quadrant[0], offset_y_mm + best_quadrant[1]
            
            # Fallback: if no good quadrant found, use opposite corner from finger cut
            if finger_cuts:
                primary_cut = finger_cuts[0]
                cut_center_x = self._inches_to_mm(primary_cut.position_x_inches) + (primary_cut.width_mm / 2)
                cut_center_y = self._inches_to_mm(primary_cut.position_y_inches) + (primary_cut.height_mm / 2)
                
                tool_center_x = offset_x_mm + tool_width_mm / 2
                tool_center_y = offset_y_mm + tool_height_mm / 2
                
                # Place number diagonally opposite to finger cut
                if cut_center_x > tool_center_x:
                    pos_x = offset_x_mm + margin_x  # Left side
                else:
                    pos_x = offset_x_mm + tool_width_mm - margin_x  # Right side
                    
                if cut_center_y > tool_center_y:
                    pos_y = offset_y_mm + margin_y  # Bottom side
                else:
                    pos_y = offset_y_mm + tool_height_mm - margin_y  # Top side
                    
                return pos_x, pos_y
            
            # Ultimate fallback: use top-right corner
            return offset_x_mm + tool_width_mm - margin_x, offset_y_mm + tool_height_mm - margin_y
            
        except Exception as e:
            logger.warning(f"Smart position calculation failed: {e}")
            # Fallback to center
            return offset_x_mm + tool_width_mm / 2, offset_y_mm + tool_height_mm / 2
            

    def _rotate_entity_manually(self, entity, degrees: float, center_x: float, center_y: float):
        """Manually rotate entity around a center point"""
        try:
            import math
            angle_rad = math.radians(degrees)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            def rotate_point(x, y):
                # Translate to origin
                x -= center_x
                y -= center_y
                # Rotate
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
                # Translate back
                return new_x + center_x, new_y + center_y
            
            if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                start_x, start_y = rotate_point(entity.dxf.start.x, entity.dxf.start.y)
                end_x, end_y = rotate_point(entity.dxf.end.x, entity.dxf.end.y)
                entity.dxf.start = (start_x, start_y, entity.dxf.start.z)
                entity.dxf.end = (end_x, end_y, entity.dxf.end.z)
            elif hasattr(entity.dxf, 'center'):
                cx, cy = rotate_point(entity.dxf.center.x, entity.dxf.center.y)
                entity.dxf.center = (cx, cy, entity.dxf.center.z)
            elif hasattr(entity, 'vertices'):
                entity_type = type(entity).__name__
                if entity_type == 'LWPolyline':
                    current_vertices = list(entity.vertices())
                    new_vertices = []
                    for vertex in current_vertices:
                        if len(vertex) >= 2:
                            new_x, new_y = rotate_point(vertex[0], vertex[1])
                            new_vertices.append((new_x, new_y))
                    entity.clear()
                    for vertex in new_vertices:
                        entity.append(vertex[:2])
        except Exception as e:
            logger.warning(f"Manual rotation failed: {e}")
            
    def _sanitize_layer_name(self, name: str, max_length: int = 50) -> str:
        r"""
        Sanitize string for use as DXF layer name.
        DXF layer names cannot contain: / \ : * ? " < > | and some other special chars
        """
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        
        # Remove or replace invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        # Remove any remaining non-alphanumeric characters except underscore and hyphen
        name = ''.join(c for c in name if c.isalnum() or c in ('_', '-'))
        
        # Ensure it doesn't start with a number (DXF requirement)
        if name and name[0].isdigit():
            name = 'T' + name
        
        # Limit length
        name = name[:max_length]
        
        # Ensure it's not empty
        if not name:
            name = 'LAYER'
        
        return name
    
    def compose_canvas_from_json(self, layout_json: Dict, output_filename: str = None, upload_to_s3: bool = True) -> Dict:
        start_time = time.time()
        
        try:
            canvas, metadata, tools = self.parse_layout_json(layout_json)
            
            if not tools:
                raise ValueError("No tools found")
            
            if not output_filename:
                safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in metadata.layout_name)
                safe_name = safe_name[:100]  # Limit length
                if not safe_name:
                    safe_name = "layout"
                output_filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.dxf"
            
            # Separate DXF tools from custom shapes
            dxf_tools = [t for t in tools if not t.is_custom_shape and t.shape_type != "fingercut"]
            shape_tools = [t for t in tools if t.is_custom_shape and t.shape_type != "fingercut"]
            fingercut_tools = [t for t in tools if t.shape_type == "fingercut"]
            
            # Download DXF files (only for non-custom tools)
            processed_tools = []
            
            # Process DXF tools with 1-inch expansion scaling
            if dxf_tools:
                download_tasks = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    for tool in dxf_tools:
                        future = executor.submit(self._download_dxf_file, tool.dxf_link, tool.tool_id)
                        download_tasks.append((tool, future))
                
                for tool, future in download_tasks:
                    local_path = future.result(timeout=60)
                    if not local_path:
                        raise Exception(f"Failed to download: {tool.name}")
                    
                    width_mm, height_mm, depth_mm, normalized_entities = self._analyze_and_normalize_dxf(local_path)
                    if not normalized_entities:
                        raise Exception(f"No entities: {tool.name}")
                    
                    # CRITICAL: Validate dimensions immediately after analysis
                    if width_mm == 0 or height_mm == 0:
                        logger.error(f"Tool '{tool.name}' has invalid dimensions: {width_mm}mm x {height_mm}mm")
                        raise Exception(f"Tool '{tool.name}' has zero width or height - invalid DXF file")
                    
                    # === STEP 1: Scale to user-defined size if needed ===
                    # normalized_segments are numpy (N,2) arrays from _analyze_and_normalize_dxf
                    if tool.height_diagonal_inches > 0:
                        expected_height_mm = self._inches_to_mm(tool.height_diagonal_inches)
                        if height_mm > 0 and abs(height_mm - expected_height_mm) > 2.0:
                            scale_factor = expected_height_mm / height_mm
                            logger.info(f"Tool '{tool.name}' scaling to expected size: "
                                        f"{height_mm:.2f}mm -> {expected_height_mm:.2f}mm  "
                                        f"(factor {scale_factor:.4f})")
                            normalized_entities = [seg * scale_factor for seg in normalized_entities]
                            width_mm  = width_mm  * scale_factor
                            height_mm = expected_height_mm

                    logger.info(f"Tool '{tool.name}': final size = {width_mm:.2f}mm x {height_mm:.2f}mm  "
                                f"segments={len(normalized_entities)}")

                                        # Store processed tool
                    tool.width_mm = width_mm
                    tool.height_mm = height_mm
                    tool.depth_mm = depth_mm
                    tool.entities = normalized_entities
                    processed_tools.append(tool)

    
            # Process custom shapes
            for tool in shape_tools + fingercut_tools:
                # Use exact dimensions from shape_data
                width_mm, height_mm, depth_mm, entities = self._create_shape_entities(
                    tool.shape_type,  # Use the actual shape type (circle, rectangle, fingercut)
                    tool.shape_data
                )
                
                if not entities:
                    logger.warning(f"Failed to create {tool.shape_type}: {tool.name}")
                    continue
                
                tool.width_mm = width_mm
                tool.height_mm = height_mm
                tool.depth_mm = depth_mm
                tool.entities = entities
                processed_tools.append(tool)
                        
            # Create DXF
            doc = ezdxf.new(units=ezdxf.units.MM)
            doc.header["$INSUNITS"] = ezdxf.units.MM
            msp = doc.modelspace()
            
            self._add_3d_canvas_representation(msp, canvas)
            
            # placed_tools = []
            # for i, tool in enumerate(processed_tools):
            #     # Apply position offsets ONLY to DXF tools (not custom shapes or fingercuts)
            #     if not tool.is_custom_shape and tool.shape_type != "fingercut":
            #         POSITION_OFFSET_INCHES = 0.4
            #         VERTICAL_OFFSET_INCHES = 0.4
                    
            #         # IMPORTANT: NO position compensation needed when scaling from center!
            #         # The tool stays in the same position relative to its center
            #         adjusted_x_inches = tool.position_x_inches + POSITION_OFFSET_INCHES
            #         adjusted_y_inches = tool.position_y_inches + VERTICAL_OFFSET_INCHES
            #     else:
            #         # No offset for custom shapes and fingercuts
            #         adjusted_x_inches = tool.position_x_inches
            #         adjusted_y_inches = tool.position_y_inches
                
            #     offset_x_mm = self._inches_to_mm(adjusted_x_inches)
            #     offset_y_mm = self._inches_to_mm(adjusted_y_inches)
            #     offset_z_mm = self._inches_to_mm(tool.position_z_inches)  # ADD THIS LINE
            #     cut_depth_mm = self._inches_to_mm(tool.cut_depth_inches)  # ADD THIS LINE
            placed_tools = []
            for i, tool in enumerate(processed_tools):
                # Apply position offsets ONLY to DXF tools (not custom shapes or fingercuts)
                if not tool.is_custom_shape and tool.shape_type != "fingercut":
                    POSITION_OFFSET_INCHES = 0.4
                    VERTICAL_OFFSET_INCHES = 0.4
                    
                    adjusted_x_inches = tool.position_x_inches + POSITION_OFFSET_INCHES
                    adjusted_y_inches = tool.position_y_inches + VERTICAL_OFFSET_INCHES
                else:
                    # No offset for custom shapes and fingercuts
                    adjusted_x_inches = tool.position_x_inches
                    adjusted_y_inches = tool.position_y_inches
                
                offset_x_mm = self._inches_to_mm(adjusted_x_inches)
                offset_y_mm = self._inches_to_mm(adjusted_y_inches)
                offset_z_mm = self._inches_to_mm(tool.position_z_inches)
                cut_depth_mm = self._inches_to_mm(tool.cut_depth_inches)
                    
                
                # CREATE LAYER NAME WITH DEPTH INFO
                if tool.is_custom_shape:
                    if tool.shape_type == "text":
                        base_layer = f"TEXT_{i+1}"
                    else:
                        base_layer = f"CUSTOM_{tool.shape_type.upper()}_{i+1}"
                else:
                    safe_name = self._sanitize_layer_name(tool.name, max_length=30)
                    base_layer = f"TOOL_{i+1}_{safe_name}"
                
                # Don't add depth suffix for text boxes
                if tool.shape_type == "text":
                    layer_name = base_layer
                else:
                    layer_name = f"{base_layer}_D{cut_depth_mm:.1f}MM"
                
                # entities_added = 0
                # for entity in tool.entities:
                #     try:
                #         new_entity = entity.copy()
                #         new_entity.dxf.layer = layer_name
                #         # Tool center in its local coordinate system (normalized to 0,0)
                #         tool_center_x = tool.width_mm / 2
                #         tool_center_y = tool.height_mm / 2
                        
                #         # ===== TRANSFORMATION ORDER: FLIP → ROTATE → TRANSLATE =====
                        
                #         # STEP 1: Apply FLIPS first (around tool center)
                #         if tool.flip_horizontal or tool.flip_vertical:
                #             # Translate to origin for flipping
                #             if hasattr(new_entity, 'translate'):
                #                 new_entity.translate(-tool_center_x, -tool_center_y, 0)
                #             else:
                #                 self._translate_entity_manually(new_entity, -tool_center_x, -tool_center_y, 0)
                            
                #             # Apply horizontal flip
                #             if tool.flip_horizontal:
                #                 self._flip_entity_horizontal(new_entity, 0)  # Flip around x=0 (now at origin)
                            
                #             # Apply vertical flip
                #             if tool.flip_vertical:
                #                 self._flip_entity_vertical(new_entity, 0)  # Flip around y=0 (now at origin)
                            
                #             # Translate back to tool space
                #             if hasattr(new_entity, 'translate'):
                #                 new_entity.translate(tool_center_x, tool_center_y, 0)
                #             else:
                #                 self._translate_entity_manually(new_entity, tool_center_x, tool_center_y, 0)
                        
                #         # STEP 2: Apply ROTATION (around tool center)
                #         if tool.rotation_degrees != 0:
                #             import math
                            
                #             # Negate rotation to fix DXF coordinate system vs canvas coordinate system
                #             corrected_rotation = -tool.rotation_degrees
                            
                #             # Translate to origin for rotation
                #             if hasattr(new_entity, 'translate'):
                #                 new_entity.translate(-tool_center_x, -tool_center_y, 0)
                #             else:
                #                 self._translate_entity_manually(new_entity, -tool_center_x, -tool_center_y, 0)
                            
                #             # Rotate around origin
                #             if hasattr(new_entity, 'rotate_z'):
                #                 new_entity.rotate_z(math.radians(corrected_rotation))
                #             else:
                #                 self._rotate_entity_manually(new_entity, corrected_rotation, 0, 0)
                            
                #             # Translate back to tool space
                #             if hasattr(new_entity, 'translate'):
                #                 new_entity.translate(tool_center_x, tool_center_y, 0)
                #             else:
                #                 self._translate_entity_manually(new_entity, tool_center_x, tool_center_y, 0)
                        
                #         # STEP 3: TRANSLATE to final canvas position
                #         if hasattr(new_entity, 'translate'):
                #             new_entity.translate(offset_x_mm, offset_y_mm, offset_z_mm)
                #         else:
                #             self._translate_entity_manually(new_entity, offset_x_mm, offset_y_mm, offset_z_mm)
                        
                #         #-----------------------------------------------------------------------------------------------------------------
                                                
                        
                #         # SET DXF ATTRIBUTES FOR CNC MACHINE
                #         new_entity.dxf.layer = layer_name
                        
                #         # ADD ELEVATION (Z-height where cutting starts)
                #         if hasattr(new_entity.dxf, 'elevation'):
                #             new_entity.dxf.elevation = offset_z_mm
                        
                #         # ADD THICKNESS (cutting depth - extrusion direction)
                #         if hasattr(new_entity.dxf, 'thickness'):
                #             new_entity.dxf.thickness = cut_depth_mm
                        
                #         new_entity.dxf.color = 1  # Red for all tool cutouts
                        
                #         msp.add_entity(new_entity)
                #         entities_added += 1
                #     except Exception as e:
                #         logger.warning(f"Entity add failed: {e}")
                #         continue
                entities_added = 0

                if not tool.is_custom_shape and tool.shape_type != "fingercut":
                    # DXF tool: pure-math transform — same quality as image-to-dxf
                    entities_added = self._place_lwpolyline_tool(
                        msp, tool, layer_name,
                        offset_x_mm, offset_y_mm, offset_z_mm, cut_depth_mm
                    )
                else:
                    # Custom shape / fingercut: original entity path
                    for entity in tool.entities:
                        try:
                            new_entity = entity.copy()
                            new_entity.dxf.layer = layer_name
                            tool_center_x = tool.width_mm / 2
                            tool_center_y = tool.height_mm / 2

                            if tool.flip_horizontal or tool.flip_vertical:
                                if hasattr(new_entity, 'translate'):
                                    new_entity.translate(-tool_center_x, -tool_center_y, 0)
                                else:
                                    self._translate_entity_manually(new_entity, -tool_center_x, -tool_center_y, 0)
                                if tool.flip_horizontal:
                                    self._flip_entity_horizontal(new_entity, 0)
                                if tool.flip_vertical:
                                    self._flip_entity_vertical(new_entity, 0)
                                if hasattr(new_entity, 'translate'):
                                    new_entity.translate(tool_center_x, tool_center_y, 0)
                                else:
                                    self._translate_entity_manually(new_entity, tool_center_x, tool_center_y, 0)

                            if tool.rotation_degrees != 0:
                                import math
                                corrected_rotation = -tool.rotation_degrees
                                if hasattr(new_entity, 'translate'):
                                    new_entity.translate(-tool_center_x, -tool_center_y, 0)
                                else:
                                    self._translate_entity_manually(new_entity, -tool_center_x, -tool_center_y, 0)
                                if hasattr(new_entity, 'rotate_z'):
                                    new_entity.rotate_z(math.radians(corrected_rotation))
                                else:
                                    self._rotate_entity_manually(new_entity, corrected_rotation, 0, 0)
                                if hasattr(new_entity, 'translate'):
                                    new_entity.translate(tool_center_x, tool_center_y, 0)
                                else:
                                    self._translate_entity_manually(new_entity, tool_center_x, tool_center_y, 0)

                            if hasattr(new_entity, 'translate'):
                                new_entity.translate(offset_x_mm, offset_y_mm, offset_z_mm)
                            else:
                                self._translate_entity_manually(new_entity, offset_x_mm, offset_y_mm, offset_z_mm)

                            new_entity.dxf.layer = layer_name
                            if hasattr(new_entity.dxf, 'elevation'):
                                new_entity.dxf.elevation = offset_z_mm
                            if hasattr(new_entity.dxf, 'thickness'):
                                new_entity.dxf.thickness = cut_depth_mm
                            new_entity.dxf.color = 1
                            msp.add_entity(new_entity)
                            entities_added += 1
                        except Exception as e:
                            logger.warning(f"Entity add failed: {e}")
                            continue
                
                placed_tools.append({
                    "tool_id": tool.tool_id,
                    "name": tool.name,
                    "type": "custom_shape" if tool.is_custom_shape else "dxf_tool",
                    "shape_type": tool.shape_type if tool.is_custom_shape else None,
                    "entities_count": entities_added,
                    "layer": layer_name,
                    "z_position_mm": offset_z_mm,
                    "cut_depth_mm": cut_depth_mm,
                    "cut_type": tool.cut_type,
                    "transformations_applied": {
                        "flip_horizontal": tool.flip_horizontal,
                        "flip_vertical": tool.flip_vertical,
                        "rotation_degrees": tool.rotation_degrees
                    }
                })

            self._add_cutting_metadata(msp, canvas, processed_tools)
            # self._add_tool_labels(msp, processed_tools) 
            self._add_tool_numbers_inside(msp, processed_tools) 
            
            output_dir = os.path.join(self.cache_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            local_output_path = os.path.join(output_dir, output_filename)
            doc.saveas(local_output_path)
            
            if not os.path.exists(local_output_path):
                raise Exception(f"Composed DXF not found at {local_output_path}")
            logger.info(f"Composed DXF ready: {local_output_path} ({os.path.getsize(local_output_path)} bytes)")

            s3_url = None
            if upload_to_s3:
                s3_url = self.s3_manager.upload_file(local_output_path, output_filename, 'application/dxf')
                if not s3_url:
                    raise Exception("S3 upload failed — check AWS credentials, bucket name, and IAM s3:PutObject permission")
            logger.info(f"Compose complete — s3_url={s3_url}")

            return {
                "success": True,
                "s3_url": s3_url,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "tools_placed": placed_tools,
                "total_tools": len(tools),
                "total_dxf_tools": len(dxf_tools),
                "total_custom_shapes": len(shape_tools),
                "total_entities": sum(t["entities_count"] for t in placed_tools)
            }
        except Exception as e:
            logger.error(f"Composition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": None,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        finally:
            self._cleanup_cache()

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/test/', methods=['GET'])
def index():
    return jsonify({
        "service": "DXF Canvas Composer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "compose": "/api/compose",
            "status": "/api/status",
            "tool_offset": "/api/tool-offset (GET/POST)", 
            "image_to_dxf": "/api/image-to-dxf"
        }
    })

@app.route('/test/api/health', methods=['GET'])
def api_health():
    try:
        s3_manager = S3Manager()
        return jsonify({
            "status": "healthy",
            "s3_available": s3_manager.s3_client is not None
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/test/api/compose', methods=['POST'])
def api_compose():
    try:
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        if "canvas_information" not in request_data:
            return jsonify({"success": False, "error": "canvas_information required"}), 400
        
        if "tools" not in request_data or not request_data["tools"]:
            return jsonify({"success": False, "error": "tools array required"}), 400
        
        # Updated validation to handle custom shapes
        for i, tool in enumerate(request_data["tools"]):
            is_custom = tool.get("is_custom_shape", False)
            
            if is_custom:
                # Validate custom shape
                if "shape_type" not in tool:
                    return jsonify({"success": False, "error": f"Tool {i}: shape_type required for custom shapes"}), 400
                if "shape_data" not in tool:
                    return jsonify({"success": False, "error": f"Tool {i}: shape_data required for custom shapes"}), 400
            else:
                # Validate DXF tool
                if "dxf_link" not in tool or not tool["dxf_link"]:
                    return jsonify({"success": False, "error": f"Tool {i} missing dxf_link"}), 400
        
        logger.info(f"Processing request: {len(request_data.get('tools', []))} tools")
        
        composer = DXFCanvasComposer()
        result = composer.compose_canvas_from_json(
            layout_json=request_data,
            output_filename=request_data.get("output_filename"),
            upload_to_s3=request_data.get("upload_to_s3", True)
        )
        
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e), "s3_url": None}), 500


from scipy.ndimage import gaussian_filter1d

# @app.route('/api/image-to-dxf', methods=['POST'])
# def api_image_to_dxf():
#     """
#     Convert cutout tool image to DXF file with contours
    
#     Expected form data:
#     - image: file (PNG/JPG with transparent or white background)
#     - length: float (tool length in inches)
#     """
#     try:
#         # Validate request
#         if 'image' not in request.files:
#             return jsonify({"success": False, "error": "No image file provided"}), 400
        
#         image_file = request.files['image']
#         if image_file.filename == '':
#             return jsonify({"success": False, "error": "Empty filename"}), 400
        
#         # Get parameters
#         try:
#             length_inches = float(request.form.get('length', 0))
#             depth_inches = float(request.form.get('depth', 0))
#         except ValueError:
#             return jsonify({"success": False, "error": "Invalid numeric parameters"}), 400
        
#         if length_inches <= 0:
#             return jsonify({"success": False, "error": "Length must be positive"}), 400
        
#         logger.info(f"Processing image: {image_file.filename}, {length_inches} inches")
        
#         # Read image
#         image_bytes = image_file.read()
#         image_bytes = compress_image_if_needed(image_bytes, max_size_mb=2)
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
#--------------------------------------------------------------------
def calculate_angle(p1, p2, p3):
    """Calculate angle at point p2 between p1-p2-p3 in degrees"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Avoid division by zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 180.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def remove_close_points(contour, min_distance=2.0):
    """Remove points that are too close together"""
    if len(contour) < 3:
        return contour
    
    filtered = [contour[0]]
    for i in range(1, len(contour)):
        dist = np.linalg.norm(contour[i] - filtered[-1])
        if dist >= min_distance:
            filtered.append(contour[i])
    
    return np.array(filtered)

# def smooth_contour_preserve_corners(contour, angle_threshold=10.0, smooth_window=3):
#     """
#     Smooth contour while preserving sharp corners.
    
#     Parameters:
#     - contour: Input contour points (Nx2 array)
#     - angle_threshold: Angles above this (in degrees) are considered sharp corners
#     - smooth_window: Window size for moving average on smooth segments
    
#     Returns:
#     - Smoothed contour with preserved sharp corners
#     """
#     if len(contour) < 3:
#         return contour
    
#     contour = contour.reshape(-1, 2).astype(np.float64)
#     n = len(contour)
    
#     # Calculate angles at each point
#     angles = []
#     for i in range(n):
#         p1 = contour[(i - 1) % n]
#         p2 = contour[i]
#         p3 = contour[(i + 1) % n]
#         angle = calculate_angle(p1, p2, p3)
#         angles.append(angle)
    
#     # Identify sharp corners (angles significantly different from 180°)
#     is_corner = np.array([abs(180 - angle) > angle_threshold for angle in angles])
    
#     # Create smoothed contour
#     smoothed = contour.copy()
    
#     # Apply moving average only to non-corner points
#     for i in range(n):
#         if not is_corner[i]:
#             # Get neighboring points (excluding corners)
#             neighbors = []
#             for offset in range(-smooth_window, smooth_window + 1):
#                 idx = (i + offset) % n
#                 if not is_corner[idx]:
#                     neighbors.append(contour[idx])
            
#             if len(neighbors) > 1:
#                 smoothed[i] = np.mean(neighbors, axis=0)
    
#     return smoothed.reshape(-1, 1, 2).astype(np.int32)

def smooth_contour_preserve_corners(contour, angle_threshold=10.0, smooth_window=3, corner_smooth_weight=0.20):
    """
    Smooth contour while preserving sharp corners.
    Now includes AGGRESSIVE straightening of straight segments to eliminate zigzag.
    
    Args:
        contour: Input contour points
        angle_threshold: Angle deviation from 180° to consider as corner
        smooth_window: Window size (unused in current implementation)
        corner_smooth_weight: Blend weight for corner smoothing (0.0-0.5)
    """
    if len(contour) < 3:
        return contour
    
    contour = contour.reshape(-1, 2).astype(np.float64)
    n = len(contour)
    
    # Calculate angles at each point
    angles = []
    for i in range(n):
        p1 = contour[(i - 1) % n]
        p2 = contour[i]
        p3 = contour[(i + 1) % n]
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)
    
    # Identify sharp corners (angles significantly different from 180°)
    is_corner = np.array([abs(180 - angle) > angle_threshold for angle in angles])
    
    # Create smoothed contour
    smoothed = contour.copy()
    
    # ========== NEW: AGGRESSIVE STRAIGHTENING OF STRAIGHT SEGMENTS ==========
    # Identify continuous straight segments (runs of non-corner points)
    # and replace them with perfectly straight lines
    
    i = 0
    while i < n:
        if not is_corner[i]:
            # Start of a straight segment - find where it ends
            segment_start = i
            
            # Find the end of this straight segment (next corner or wrap around)
            while i < n and not is_corner[i]:
                i += 1
            
            segment_end = i if i < n else n
            
            # If segment has 3+ points, straighten it completely
            segment_length = segment_end - segment_start
            if segment_length >= 3:
                # Get start and end points
                start_point = contour[segment_start]
                end_point = contour[segment_end % n]
                
                # LINEAR INTERPOLATION: create perfectly straight line
                for j in range(segment_start + 1, segment_end):
                    t = (j - segment_start) / (segment_end - segment_start)
                    smoothed[j] = start_point * (1 - t) + end_point * t
        else:
            i += 1
    
    # ========== MICRO-LEVEL CORNER SMOOTHING ==========
    # Apply smoothing ONLY at corner points for better CNC performance
    
    if corner_smooth_weight > 0.0:  # Skip if no smoothing requested
        corner_smooth_radius = 1  # Just immediate neighbors
        
        for i in range(n):
            if is_corner[i]:
                # Get immediate neighbors (just 1 point on each side)
                p_prev = smoothed[(i - corner_smooth_radius) % n]
                p_curr = smoothed[i]
                p_next = smoothed[(i + corner_smooth_radius) % n]
                
                # Apply weighted average - creates smooth micro-radius at corner
                smoothed[i] = (
                    (1 - 2 * corner_smooth_weight) * p_curr +
                    corner_smooth_weight * p_prev +
                    corner_smooth_weight * p_next
                )
    # ========== END SMOOTHING ==========
    
    return smoothed.reshape(-1, 1, 2).astype(np.int32)


def adaptive_smooth_contour(contour, preserve_sharp_angles=True, corner_smooth_weight=0.20):
    """
    Multi-stage adaptive smoothing that preserves geometry while removing zig-zags.
    Tuned for MINIMAL smoothing - just enough to remove pixel-level noise.
    
    Args:
        contour: Input contour points
        preserve_sharp_angles: If True, apply corner smoothing
        corner_smooth_weight: Smoothing intensity (0.0-0.5, default 0.20)
    """
    # Stage 1: Remove points that are too close (< 1.5 pixels apart)
    points = contour.reshape(-1, 2)
    points = remove_close_points(points, min_distance=1.0)  # Very tight - only removes duplicates
    
    # Stage 2: Micro Douglas-Peucker - removes only exact duplicate points
    contour_temp = points.reshape(-1, 1, 2).astype(np.int32)
    epsilon_micro = 0.05  # EXTREMELY small - only removes exact duplicates
    contour_temp = cv2.approxPolyDP(contour_temp, epsilon_micro, True)
    
    if not preserve_sharp_angles or corner_smooth_weight == 0.0:
        return contour_temp
    
    # Stage 3: VERY light angle-based smoothing
    smoothed = smooth_contour_preserve_corners(
        contour_temp, 
        angle_threshold=8.0,              # Preserve even slight corners
        smooth_window=1,                  # Minimal window - just neighbors
        corner_smooth_weight=corner_smooth_weight  # Pass weight through
    )
    
    return smoothed


# ============================================================
#  MICRO-LEVEL SMOOTHING  (second pass, pixel-space)
# ============================================================
# Tunable config — adjust these four values to taste:
#
#   MICRO_SIGMA          : Gaussian width in points.
#                          1.0 = very subtle, 2.5 = moderate, 4.0 = strong
#                          Controls how hard zigzags are scrubbed on curves.
#                          Default 1.8 removes 1-2 px staircase teeth without
#                          visibly moving true curves.
#
#   MICRO_CORNER_ANGLE   : Deviation from 180° above which a point is a
#                          "sharp corner" → fully protected, never smoothed.
#                          Default 25° protects tool corners (rounded rect
#                          corners, chisel tips, etc.)
#
#   MICRO_STRAIGHT_ANGLE : Deviation from 180° BELOW which a point is on a
#                          "straight edge" → also fully protected.
#                          This prevents straight sides from bending inward.
#                          Default 3.0° — only points that are nearly perfectly
#                          straight get shielded. Raise to 6-8 if straight
#                          sides still bend after smoothing.
#
#   MICRO_CORNER_GUARD   : Number of points on EACH SIDE of a detected corner
#                          or straight-edge point that are also protected
#                          (tapers to zero). Default 3.
#
MICRO_SIGMA          = 2.8   # ← Gaussian sigma. Tune: 1.0–4.0
MICRO_CORNER_ANGLE   = 25.0  # ← Sharp corner threshold (deg). Tune: 15–45
MICRO_STRAIGHT_ANGLE = 6.0   # ← Straight edge threshold (deg). Tune: 1.0–8.0
MICRO_CORNER_GUARD   = 3     # ← Protected neighbours each side. Tune: 1–6

# ── Final mm-space smoothing pass (applied to dxf_points before spline) ──────
# Kills any residual micro-bumps that survived pixel-space smoothing and
# prevents the spline from creating S-waves between surviving noise points.
#
#   MM_SMOOTH_SIGMA : Gaussian sigma in DXF points (not mm).
#                     1.0 = very light,  2.0 = moderate,  3.5 = strong.
#                     Default 1.5 — smooths S-waves without shifting geometry.
#                     Set to 0.0 to disable entirely.
#
MM_SMOOTH_SIGMA = 1.5        # ← mm-space Gaussian sigma. Tune: 0.0–3.5

def remove_contour_spikes(pts_2d: np.ndarray,
                          window: int = 5,
                          z_thresh: float = 3.5) -> np.ndarray:
    """
    Surgical outlier spike removal for closed contours.

    Problem the SG filter cannot solve
    ------------------------------------
    SG (and any linear filter) treats every point equally within its window.
    A single point that has jumped 10–20 px away from the true contour pulls
    the polynomial fit toward it, leaving a visible spike in the output.
    The fix is to detect such outliers BEFORE filtering and replace them with
    a linearly interpolated estimate from their clean neighbours so the
    subsequent SG pass sees no statistical outliers at all.

    Algorithm
    ---------
    For each point i, compute its *perpendicular deviation* from the chord
    connecting its two nearest non-spike neighbours (±window points).  The
    distribution of these deviations across the whole contour is nearly
    Gaussian for a smooth tool outline; genuine spikes are far-outliers.
    We flag points whose |deviation| exceeds z_thresh × MAD-based σ and
    replace them with the linear interpolant between their flanking neighbours.

    This is done in two passes so that clusters of consecutive spike points
    (e.g. a 2–3 pt notch) are also caught.

    Parameters
    ----------
    pts_2d   : (N, 2) float64 array — contour in any coordinate space
    window   : half-width of the "good neighbour" look-ahead used for the
               chord.  5 means we look ±5 points away, skipping the
               immediate neighbours that might also be spiked.
               Raise to 7–9 if multi-point spikes still survive.
    z_thresh : how many robust-σ a deviation must exceed to be called a
               spike.  3.5 is conservative (keeps real sharp corners);
               lower to 2.5 if faint jitters remain, raise to 5.0 if real
               geometric features are being clipped.

    Returns
    -------
    (N, 2) float64 array with spikes replaced by interpolated values.
    """
    pts = pts_2d.copy().astype(np.float64)
    n   = len(pts)
    if n < 2 * window + 3:
        return pts

    def _one_pass(p):
        devs = np.zeros(n)
        for i in range(n):
            prev = p[(i - window) % n]
            nxt  = p[(i + window) % n]
            chord = nxt - prev
            chord_len = np.linalg.norm(chord)
            if chord_len < 1e-9:
                devs[i] = 0.0
                continue
            # Signed perpendicular distance from point to chord
            t = np.dot(p[i] - prev, chord) / (chord_len ** 2)
            proj = prev + t * chord
            devs[i] = np.linalg.norm(p[i] - proj)

        # Robust σ via MAD (insensitive to the very outliers we are detecting)
        med  = np.median(devs)
        mad  = np.median(np.abs(devs - med))
        sigma_robust = max(mad * 1.4826, 1e-6)   # MAD → σ scaling constant
        threshold    = med + z_thresh * sigma_robust

        spike_mask = devs > threshold
        n_spikes   = int(spike_mask.sum())
        if n_spikes == 0:
            return p, 0

        # Replace spikes with linear interpolation between nearest clean neighbours
        result = p.copy()
        for i in range(n):
            if not spike_mask[i]:
                continue
            # Walk outward to find a clean point on each side
            lo, hi = i, i
            for step in range(1, n):
                lo = (i - step) % n
                if not spike_mask[lo]:
                    break
            for step in range(1, n):
                hi = (i + step) % n
                if not spike_mask[hi]:
                    break
            # Linear interpolation along the contour index
            lo_idx = lo if lo < i else lo - n
            hi_idx = hi if hi > i else hi + n
            span   = hi_idx - lo_idx
            if span == 0:
                continue
            t = (i - lo_idx) / span
            result[i] = (1 - t) * p[lo % n] + t * p[hi % n]

        return result, n_spikes

    # Two passes: first pass fixes isolated spikes; second catches neighbours
    # that looked "clean" only because they were next to an even larger spike.
    pts, n1 = _one_pass(pts)
    pts, n2 = _one_pass(pts)
    total   = n1 + n2
    if total:
        logger.info(f"remove_contour_spikes: removed {total} spike pts "
                    f"(pass1={n1}, pass2={n2}, window={window}, z={z_thresh})")
    return pts


def remove_dxf_spikes(dxf_points: list,
                      window: int = 5,
                      z_thresh: float = 3.5) -> list:
    """
    Wrapper around remove_contour_spikes for mm-space DXF point lists.
    Accepts / returns list of (x, y) tuples.
    """
    if len(dxf_points) < 2 * window + 3:
        return dxf_points
    pts = np.array(dxf_points, dtype=np.float64)
    pts = remove_contour_spikes(pts, window=window, z_thresh=z_thresh)
    return [(float(x), float(y)) for x, y in pts]


def _sg_pass(pts: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Single SG pass on a closed (N,2) contour with circular wrap padding."""
    pad   = window
    x_pad = np.concatenate([pts[-pad:, 0], pts[:, 0], pts[:pad, 0]])
    y_pad = np.concatenate([pts[-pad:, 1], pts[:, 1], pts[:pad, 1]])
    xs    = savgol_filter(x_pad, window, poly)[pad:-pad]
    ys    = savgol_filter(y_pad, window, poly)[pad:-pad]
    return np.stack([xs, ys], axis=1)


def _kill_curvature_spikes(pts: np.ndarray,
                            curv_z_thresh: float = 3.0,
                            guard: int = 2) -> np.ndarray:
    """
    Detect and flatten sub-millimetre curvature spikes that survive SG.

    Root cause
    ----------
    SG's weight-blend boundary (where the corner-guard taper meets fully-
    smoothed points) creates a tiny discontinuity in the first derivative.
    At 0.2 mm scale this shows as a sharp V-notch or hook.  These spikes
    have NORMAL positional amplitude so the outlier detector misses them,
    but they have EXTREME local curvature — geometrically impossible for
    a real machined surface.

    Method
    ------
    1. Compute discrete signed curvature κ at every point.
    2. Estimate background distribution robustly (MAD-σ).
    3. Flag points whose |κ| > curv_z_thresh × σ above median.
    4. Replace with linear interpolation + narrow Hann blend over guard zone.
    """
    pts = pts.copy()
    n   = len(pts)
    if n < 7:
        return pts

    kappa = np.zeros(n)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        d1     = p1 - p0
        d2     = p2 - p1
        cross  = d1[0] * d2[1] - d1[1] * d2[0]
        norm3  = np.linalg.norm(d1) ** 3
        kappa[i] = cross / norm3 if norm3 > 1e-12 else 0.0

    abs_k  = np.abs(kappa)
    med_k  = np.median(abs_k)
    mad_k  = np.median(np.abs(abs_k - med_k))
    sig_k  = max(mad_k * 1.4826, 1e-9)
    thresh = med_k + curv_z_thresh * sig_k

    spike_mask = abs_k > thresh
    n_spikes   = int(spike_mask.sum())
    if n_spikes == 0:
        return pts

    repaired = pts.copy()
    for i in range(n):
        if not spike_mask[i]:
            continue
        lo, hi = i, i
        for s in range(1, n):
            lo = (i - s) % n
            if not spike_mask[lo]:
                break
        for s in range(1, n):
            hi = (i + s) % n
            if not spike_mask[hi]:
                break
        lo_i = lo if lo < i else lo - n
        hi_i = hi if hi > i else hi + n
        span = hi_i - lo_i
        if span == 0:
            continue
        t = (i - lo_i) / span
        repaired[i] = (1 - t) * pts[lo % n] + t * pts[hi % n]

    blend_mask = spike_mask.copy()
    for i in range(n):
        if spike_mask[i]:
            for g in range(1, guard + 1):
                blend_mask[(i - g) % n] = True
                blend_mask[(i + g) % n] = True

    for i in range(n):
        if blend_mask[i]:
            p0 = repaired[(i - 1) % n]
            p1 = repaired[i]
            p2 = repaired[(i + 1) % n]
            repaired[i] = 0.25 * p0 + 0.50 * p1 + 0.25 * p2

    logger.info(
        f"_kill_curvature_spikes: {n_spikes} curvature spikes removed "
        f"(thresh={thresh:.5f}, med_κ={med_k:.5f}, σ={sig_k:.5f})"
    )
    return repaired


def micro_smooth_contour_pixels(contour,
                                sigma=None,           # kept for API compatibility; unused
                                corner_angle_threshold=MICRO_CORNER_ANGLE,
                                straight_angle_threshold=MICRO_STRAIGHT_ANGLE,
                                corner_guard=MICRO_CORNER_GUARD):
    """
    Micro-level Savitzky-Golay smoothing in pixel space.

    Replaces the previous Gaussian pass.  SG fits a local polynomial to each
    window, so it preserves true curvature (curves stay curved, straights stay
    straight) while killing the zero-mean high-frequency zigzag that gaussian
    smoothing can only attenuate.

    Tuning
    ------
    SG_WINDOW : odd integer ≥ SG_POLY+2.  Larger → more smoothing.
                11 pts covers ~5–6 px of context at typical contour density.
                Raise to 15–21 if roughness persists; lower to 7 if real
                geometry starts rounding off.
    SG_POLY   : polynomial order.  3 (cubic) fits curves well without
                over-fitting noise.  Do not exceed SG_WINDOW-2.

    Zone logic (unchanged from Gaussian version)
    -------------------------------------------
    • deviation > corner_angle_threshold  → sharp corner   → weight = 0
    • deviation < straight_angle_threshold → straight edge  → weight = 0
    • everything in between               → curve segment  → weight = 1
    Points adjacent to any protected point are tapered via corner_guard.
    """
    SG_WINDOW = 11   # ← Tune: 7 / 11 / 15 / 21 (must be odd)
    SG_POLY   = 3    # ← Tune: 2 or 3 (cubic recommended)

    if len(contour) < SG_WINDOW + 2:
        return contour

    pts = contour.reshape(-1, 2).astype(np.float64)
    n   = len(pts)

    # ── 1. Compute deviation-from-straight at every point ───────────────────
    deviations = np.zeros(n)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v1 = p0 - p1
        v2 = p2 - p1
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            deviations[i] = 0.0
            continue
        cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        deviations[i] = abs(180.0 - np.degrees(np.arccos(cos_a)))

    # ── 2. Classify every point ──────────────────────────────────────────────
    is_sharp_corner  = deviations > corner_angle_threshold
    is_straight_edge = deviations < straight_angle_threshold
    is_protected     = is_sharp_corner | is_straight_edge

    # ── 3. Build smooth-blend weight with taper around protected points ──────
    weight = np.ones(n, dtype=np.float64)
    for i in range(n):
        if is_protected[i]:
            weight[i] = 0.0
            for k in range(1, corner_guard + 1):
                taper = 1.0 - (k / (corner_guard + 1))
                weight[(i - k) % n] = min(weight[(i - k) % n], taper)
                weight[(i + k) % n] = min(weight[(i + k) % n], taper)

    # ── 4. SG smooth with wrap-around padding (closed contour) ──────────────
    pad   = SG_WINDOW
    x_raw = pts[:, 0]
    y_raw = pts[:, 1]
    x_pad = np.concatenate([x_raw[-pad:], x_raw, x_raw[:pad]])
    y_pad = np.concatenate([y_raw[-pad:], y_raw, y_raw[:pad]])
    xs_smooth = savgol_filter(x_pad, SG_WINDOW, SG_POLY)[pad:-pad]
    ys_smooth = savgol_filter(y_pad, SG_WINDOW, SG_POLY)[pad:-pad]

    # ── 5. Blend: result = weight * smoothed + (1-weight) * original ────────
    xs_out = weight * xs_smooth + (1.0 - weight) * x_raw
    ys_out = weight * ys_smooth + (1.0 - weight) * y_raw

    # ── 6. Curvature spike kill at blend boundaries ───────────────────────────
    # The weight-blend taper (step 3) creates a tiny first-derivative
    # discontinuity right where the protected zone meets the smoothed zone.
    # At sub-pixel scale this shows as a 0.2–0.5 px hook/notch.
    # Kill it here before the result is cast back to int32.
    blended = np.stack([xs_out, ys_out], axis=1)
    blended = _kill_curvature_spikes(blended, curv_z_thresh=3.0, guard=2)
    xs_out  = blended[:, 0]
    ys_out  = blended[:, 1]

    logger.info(
        f"micro_smooth_contour_pixels [SG w={SG_WINDOW} p={SG_POLY}]: "
        f"n={n}, sharp_corners={int(is_sharp_corner.sum())}, "
        f"straight_pts={int(is_straight_edge.sum())}, "
        f"smoothed_pts={int((weight > 0.5).sum())}"
    )

    result = np.stack([xs_out, ys_out], axis=1)
    return result.reshape(-1, 1, 2).astype(np.int32)
# ============================================================


def _rolling_smooth_pass(pts: np.ndarray, window: int, blend: float) -> np.ndarray:
    """
    Single pass of window-rolling average blend on a closed contour (Nx2 array).
    Each point is blended toward the mean of its ±window neighbours.
    Uses vectorised circular indexing — O(N) not O(N*window).
    """
    n   = len(pts)
    acc = np.zeros_like(pts)
    k   = 2 * window + 1
    # Build circular sum via cumsum on tiled array
    tiled = np.tile(pts, (3, 1))          # 3× tiled so we can slice without modulo
    cs    = np.cumsum(tiled, axis=0)
    start = n                              # offset into middle tile
    for i in range(n):
        lo = start + i - window
        hi = start + i + window + 1
        acc[i] = (cs[hi] - cs[lo]) / k
    return (1.0 - blend) * pts + blend * acc




def smooth_dxf_points_mm(dxf_points):
    """
    Full mm-space smoothing pipeline — 4 stages:

    Stage 1 — Coarse SG (window=31, cubic)
        Flattens the large-scale rasterisation waviness that makes the
        contour look bumpy at the 1–5 mm scale.

    Stage 2 — Curvature spike killer
        Detects points with pathological local curvature (geometrically
        impossible for a real machined surface) and replaces them with
        interpolated values.  This catches the 0.2 mm V-notches and hooks
        that SG's weight-blend boundary can create.

    Stage 3 — Fine SG (window=15, cubic)
        Re-smooths the repair edges from stage 2 and eliminates any
        residual micro-waviness the coarse pass missed.

    Stage 4 — Second curvature pass (tighter threshold)
        Catches any new micro-spikes introduced at stage-3 repair boundaries.
        Using a tighter z_thresh=2.5 here is safe because by this point the
        background curvature is already well-behaved.

    Why not just one giant SG window?
        A single wide SG window would round off real geometric features
        (shoulder curves, petal tips).  The staged approach keeps each
        individual step small so geometry is preserved throughout.
    """
    SG1_W, SG1_P = 31, 3   # Stage 1: coarse SG  (odd, ≥ poly+2)
    SG2_W, SG2_P = 15, 3   # Stage 3: fine SG    (odd, ≥ poly+2)
    MIN_PTS      = SG1_W + 2

    if len(dxf_points) < MIN_PTS:
        return dxf_points

    pts = np.array(dxf_points, dtype=np.float64)

    # Stage 1 — coarse SG
    pts = _sg_pass(pts, SG1_W, SG1_P)

    # Stage 2 — curvature spike killer (catches 0.2 mm micro-jitters)
    pts = _kill_curvature_spikes(pts, curv_z_thresh=3.0, guard=2)

    # Stage 3 — fine SG to smooth repair edges
    if len(pts) >= SG2_W + 2:
        pts = _sg_pass(pts, SG2_W, SG2_P)

    # Stage 4 — second curvature pass with tighter threshold
    pts = _kill_curvature_spikes(pts, curv_z_thresh=2.5, guard=1)

    logger.info(
        f"smooth_dxf_points_mm: 4-stage pipeline complete, {len(pts)} pts"
    )

    return [(float(pts[i, 0]), float(pts[i, 1])) for i in range(len(pts))]

#---------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER  —  image compression
# ─────────────────────────────────────────────────────────────────────────────

def _compress_for_pipeline(image_bytes, max_size_mb=2):
    """Compress image bytes to <= max_size_mb."""
    max_bytes = max_size_mb * 1024 * 1024
    if len(image_bytes) <= max_bytes:
        return image_bytes
    logger.info(f"Compressing image {len(image_bytes)/1024/1024:.2f}MB -> <={max_size_mb}MB")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return image_bytes
    quality = 95
    while quality > 10:
        _, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, quality // 10])
        if len(buf) <= max_bytes:
            return buf.tobytes()
        quality -= 10
    sf = (max_bytes / len(image_bytes)) ** 0.5
    img_small = cv2.resize(img, (int(img.shape[1]*sf), int(img.shape[0]*sf)),
                           interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode('.png', img_small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINT  —  /test/api/image-to-dxf
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/test/api/image-to-dxf', methods=['POST'])
def api_image_to_dxf():
    """
    Full pipeline endpoint — no Gradio, pure Flask.

    Uses the EXACT same core logic as the original app(13).py pipeline:
      originalImage  ->  gradio_detect_tool (YOLOv11x cascade + CV fallback)
                     ->  seg_sam (2-pass shadow removal)
                     ->  run_single_model_pipeline  ->  DXF + contour
                     ->  S3 upload  ->  JSON response

    Input (multipart/form-data):
      originalImage   : file   — the only image used by the pipeline
      segmentedImage  : file   — accepted, completely ignored (for API compat)
      lengthInches    : float  — tool length in inches  (preferred)
        OR  length + unit      — length + 'inches'/'mm'
      depthInches     : float  — tool depth in inches
        OR  depth  + unit
      toolType / toolBrand / SKUorPartNumber : str (logged only)

    Response JSON:
      { "success", "dxf_url", "original_img", "contour_image_url" }
    """
    logger.info("=== /test/api/image-to-dxf ===")
    for k, v in request.form.items():
        logger.info(f"  form[{k}]: {v}")
    for k, f in request.files.items():
        logger.info(f"  file[{k}]: {f.filename}")

    try:
        # ── 1. Read originalImage ─────────────────────────────────────────────
        if 'originalImage' not in request.files:
            return jsonify({"success": False, "error": "Missing: 'originalImage'"}), 400

        img_file = request.files['originalImage']
        if img_file.filename == '':
            return jsonify({"success": False, "error": "Empty filename for 'originalImage'"}), 400

        # ── PIL-based decode: replicates Gradio gr.Image(type="pil") exactly ───
        # Gradio's gr.Image component automatically applies EXIF orientation
        # before passing the image to any function. In this Flask endpoint we
        # must do that step ourselves with ImageOps.exif_transpose(), otherwise
        # phone photos arrive rotated and the whole pipeline output is rotated.
        from PIL import ImageOps as _ImageOps
        import io as _io
        raw_bytes = img_file.read()
        try:
            _pil_img = Image.open(_io.BytesIO(raw_bytes))
            _pil_img = _ImageOps.exif_transpose(_pil_img)  # apply EXIF orientation (what Gradio did silently)
            _pil_img = _pil_img.convert("RGB")             # strip alpha only
            img_bgr  = cv2.cvtColor(np.array(_pil_img), cv2.COLOR_RGB2BGR)
        except Exception as _pil_err:
            logger.warning(f"PIL decode failed ({_pil_err}) — falling back to cv2")
            nparr    = np.frombuffer(raw_bytes, np.uint8)
            img_raw  = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img_raw is None:
                return jsonify({"success": False, "error": "Cannot decode 'originalImage'"}), 400
            if len(img_raw.shape) == 2:
                img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            elif img_raw.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = img_raw.copy()

        # segmentedImage: accept and immediately discard — never used by pipeline
        if 'segmentedImage' in request.files:
            logger.info("segmentedImage received but IGNORED — pipeline uses originalImage only")

        img_h, img_w = img_bgr.shape[:2]
        logger.info(f"originalImage: {img_w}x{img_h}")

        # ── 2. Parse dimensions ───────────────────────────────────────────────
        try:
            raw_len = request.form.get('lengthInches') or request.form.get('length')
            raw_dep = request.form.get('depthInches')  or request.form.get('depth')
            unit    = (request.form.get('unit') or 'inches').lower().strip()

            if not raw_len:
                return jsonify({"success": False,
                                "error": "Missing: 'lengthInches' or 'length'"}), 400

            lv = float(raw_len)
            dv = float(raw_dep) if raw_dep else 0.0

            if request.form.get('lengthInches') is not None:
                length_in, depth_in = lv, dv          # already in inches
            elif unit == 'mm':
                length_in, depth_in = lv / 25.4, dv / 25.4
            else:
                length_in, depth_in = lv, dv

        except (ValueError, TypeError) as exc:
            return jsonify({"success": False, "error": f"Bad numeric params: {exc}"}), 400

        if length_in <= 0:
            length_in = 8.0
            logger.warning("length_in <= 0 — defaulted to 8.0 in")

        logger.info(
            f"length={length_in:.3f}in  depth={depth_in:.3f}in  "
            f"tool={request.form.get('toolType','')}  "
            f"brand={request.form.get('toolBrand','')}  "
            f"sku={request.form.get('SKUorPartNumber','')}"
        )

        # ── 3. Detection — full cascade (YOLOv11x->v11m->v8x->v8m + CV fallback)
        logger.info("Step 1/3: detection (full cascade + CV fallback) ...")
        _t0 = time.time()
        # gradio_detect_tool returns: (bbox, thresh_mask, method, warning, raw_boxes)
        bbox, thresh_mask, det_method, warning, raw_boxes = gradio_detect_tool(img_bgr)
        logger.info(f"  Detection done in {time.time()-_t0:.1f}s  method={det_method}  bbox={bbox}  warning={warning is not None}")
        if warning:
            logger.warning(f"  Detection warning: {warning}")

        # ── 4. SAM Segmentation (2-pass shadow-aware, identical to Gradio flow) ─
        logger.info("Step 2/3: SAM segmentation (2-pass shadow removal) ...")
        _t0 = time.time()
        try:
            binary_mask = seg_sam(img_bgr, bbox)
        except Exception as e:
            logger.error(f"SAM failure: {e}", exc_info=True)
            return jsonify({"success": False, "error": f"SAM segmentation failed: {e}"}), 500
        logger.info(f"  SAM done in {time.time()-_t0:.1f}s  mask_px={np.count_nonzero(binary_mask)}")

        # ── 5. Contour + DXF pipeline (exactly as in Gradio flow) ────────────
        logger.info("Step 3/3: run_single_model_pipeline ...")
        res = run_single_model_pipeline(img_bgr, binary_mask, length_in, depth_in, "SAM")

        if res["dxf_path"] is None:
            return jsonify({"success": False, "error": "Pipeline failed — no DXF produced"}), 500

        dxf_path    = res["dxf_path"]
        contour_img = res["contour_img"]   # original image with contour drawn
        merged_img  = res["merged_img"]    # grey foam bg + tool + gold/grey contours

        # ── 6. Build output images (full-size, uncropped) ─────────────────────
        uid = uuid.uuid4().hex[:8]

        # original_img: full original image as received (BGR -> PNG, no crop, no alpha)
        orig_filename = f"original_{uid}.png"
        orig_path     = os.path.join(HF_CACHE_DIR, orig_filename)
        cv2.imwrite(orig_path, img_bgr)
        logger.info(f"original_img saved (full-size {img_w}x{img_h}): {orig_path}")

        # contour_image_url: full original image with contour drawn on top (no crop)
        # res["contour_img"] is already the full-size image with the orange contour
        # drawn by run_single_model_pipeline — use it directly.
        cont_filename = f"contour_{uid}.png"
        cont_path     = os.path.join(HF_CACHE_DIR, cont_filename)
        if res["contour_img"] is not None:
            cv2.imwrite(cont_path, res["contour_img"])
            logger.info(f"contour_img saved (full-size, pipeline contour): {cont_path}")
        else:
            # Fallback: draw contour manually on full image if pipeline didn't produce one
            mask_bin = binary_mask.copy()
            if len(mask_bin.shape) == 3:
                mask_bin = (cv2.cvtColor(mask_bin, cv2.COLOR_BGR2GRAY)
                            if mask_bin.shape[2] == 3 else mask_bin[:, :, 3])
            _, mask_bin = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)
            if mask_bin.shape[:2] != (img_h, img_w):
                mask_bin = cv2.resize(mask_bin, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            contour_vis = img_bgr.copy()
            mask_pad = cv2.copyMakeBorder(mask_bin, 1, 1, 1, 1,
                                           cv2.BORDER_CONSTANT, value=0)
            contours_list, _ = cv2.findContours(mask_pad, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
            if contours_list:
                ct_px  = max(2, int((0.05 / max(length_in, 1e-6)) * img_h))
                main_c = max(contours_list, key=cv2.contourArea) - 1
                cv2.drawContours(contour_vis, [main_c], -1,
                                 (0, 80, 255), ct_px, cv2.LINE_AA)
            cv2.imwrite(cont_path, contour_vis)
            logger.info(f"contour_img saved (fallback manual draw): {cont_path}")

        # ── 7. Upload to S3 ───────────────────────────────────────────────────
        dxf_filename = os.path.basename(dxf_path)
        s3 = S3Manager()
        dxf_url  = s3.upload_file(dxf_path,  dxf_filename,  'application/dxf')
        orig_url = s3.upload_file(orig_path,  orig_filename, 'image/png')
        cont_url = s3.upload_file(cont_path,  cont_filename, 'image/png')

        # Clean up temp files
        for p in [dxf_path, orig_path, cont_path]:
            try:
                if os.path.exists(p): os.remove(p)
            except Exception:
                pass

        if not all([dxf_url, orig_url, cont_url]):
            return jsonify({"success": False,
                            "error": "S3 upload failed for one or more files"}), 500

        logger.info(f"=== image-to-dxf COMPLETE ===")
        logger.info(f"  [DXF]     {dxf_url}")
        logger.info(f"  [ORIG]    {orig_url}")
        logger.info(f"  [CONTOUR] {cont_url}   <- open this to verify contour quality")
        return jsonify({
            "success":           True,
            "dxf_url":           dxf_url,
            "original_img":      orig_url,
            "contour_image_url": cont_url,
        }), 200

    except Exception as exc:
        logger.error(f"api_image_to_dxf unhandled error: {exc}", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route('/test/api/status', methods=['GET'])
def api_status():
    try:
        s3_manager = S3Manager()
        return jsonify({
            "api_version": "1.0.0",
            "status": "running",
            "s3_available": s3_manager.s3_client is not None,
            "aws_region": AWS_REGION
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test/api/tool-offset', methods=['GET', 'POST'])
def api_tool_offset():
    """Get or update tool contour offset configuration"""
    try:
        if request.method == 'GET':
            current_offset = get_tool_offset_inches()
            return jsonify({
                "success": True,
                "tool_contour_offset_inches": current_offset
            }), 200
        
        elif request.method == 'POST':
            data = request.get_json()
            if not data or 'tool_contour_offset_inches' not in data:
                return jsonify({
                    "success": False,
                    "error": "tool_contour_offset_inches required"
                }), 400
            
            new_offset = float(data['tool_contour_offset_inches'])
            
            if new_offset < 0 or new_offset > 1.0:
                return jsonify({
                    "success": False,
                    "error": "Offset must be between 0 and 1.0 inches"
                }), 400
            
            if set_tool_offset_inches(new_offset):
                return jsonify({
                    "success": True,
                    "tool_contour_offset_inches": new_offset,
                    "message": f"Tool offset updated to {new_offset} inches"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to update offset"
                }), 500
                
    except Exception as e:
        logger.error(f"Tool offset API error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
        
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": ["/test/", "/test/api/health", "/test/api/compose", "/test/api/status", "/test/api/image-to-dxf"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW: GRADIO SEGMENTATION PIPELINE  (v2 — fixed detection & segment selection)
#  All original code above is 100% preserved.
# ═══════════════════════════════════════════════════════════════════════════════

import threading
import tempfile
import warnings
warnings.filterwarnings("ignore")

GRADIO_OUTPUT_DIR = tempfile.mkdtemp()
_seg_models_cache = {}   # lazy-loaded singletons


# ═════════════════════════════════════════════════════════════════════════════
#  SMART BBOX DETECTION  — v3
#
#  Pipeline (in priority order):
#    1. YOLOv11x  (Ultralytics — latest generation, best accuracy)
#       └─ fallback → YOLOv11m  → YOLOv8x  → YOLOv8m  (automatic cascade)
#    2. Enhanced multi-method CV fallback
#       (Otsu ×2 + LAB channels + Canny fill + HSV + GrabCut refinement)
#    3. Centre-crop last resort
#
#  KEY FEATURE — Whole-tool unification
#    Many tools (screwdrivers, pliers, wrenches …) have multi-coloured grips.
#    YOLO sometimes fires several boxes for the same physical object — one for
#    the metal shaft, one for the rubber handle, etc.  We solve this by:
#      a) Collecting ALL YOLO boxes above a low confidence threshold.
#      b) Running an IoU + proximity-based merge that unions any boxes that
#         belong to the same physical object (overlapping or within a gap
#         threshold of each other).
#      c) Counting the resulting "object clusters" AFTER merging.
#         • 1 cluster  → normal flow
#         • 2+ clusters → return a warning flag so the UI can show a message
# ═════════════════════════════════════════════════════════════════════════════

# ─── Box-geometry helpers ────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _gap_distance(a: list, b: list) -> float:
    """
    Minimum pixel gap between two boxes (0 if they overlap or touch).
    Measures the closest distance between any edges.
    """
    # Horizontal gap
    hgap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    # Vertical gap
    vgap = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(hgap, vgap)   # diagonal gap ≈ max of both gaps (conservative)


def _union_box(boxes: list) -> list:
    """Return the axis-aligned bounding box that encloses all input boxes."""
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def _merge_tool_boxes(boxes: list, img_w: int, img_h: int) -> list:
    """
    Merge YOLO boxes into whole-tool clusters using gap-based Union-Find.

    Cable pre-filter: strip thin elongated boxes touching the image border.

    Clustering: two boxes belong to the same tool when EITHER:
      • IoU > 0.05  (overlapping)
      • Gap < 0.12 × min(image dimension)  (nearby parts: shaft + grip)
    Union-Find resolves transitive chains in one pass so multi-color tools
    whose parts are detected as separate boxes (blade + handle, shaft + grip)
    are unified into one tight bbox — without over-expanding to the full image.
    """
    if not boxes:
        return []

    # ── Cable pre-filter ─────────────────────────────────────────────────────
    def _box_is_cable(b) -> bool:
        x1, y1, x2, y2 = b
        bw, bh     = x2 - x1, y2 - y1
        short, lng = sorted([bw, bh])
        aspect     = lng / max(short, 1)
        touches    = (x1 <= 4 or y1 <= 4 or
                      x2 >= img_w - 4 or
                      y2 >= img_h - 4)
        return aspect > 4.5 and touches

    non_cable = [b for b in boxes if not _box_is_cable(b)]
    working   = non_cable if non_cable else boxes
    if len(non_cable) < len(boxes):
        logger.info(f"Cable pre-filter: removed {len(boxes)-len(non_cable)} cable-like box(es)")

    # ── Union-Find gap clustering ─────────────────────────────────────────────
    gap_thresh = 0.12 * min(img_w, img_h)
    n      = len(working)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if (_iou(working[i], working[j]) > 0.05 or
                    _gap_distance(working[i], working[j]) < gap_thresh):
                union(i, j)

    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(working[i])

    merged = [_union_box(grp) for grp in clusters.values()]
    logger.info(f"Box merge: {len(boxes)} raw → {len(non_cable)} non-cable → "
                f"{len(merged)} cluster(s)  (gap_thresh={gap_thresh:.1f}px)")
    return merged

# ─── CV fallback detector ────────────────────────────────────────────────────

def _is_cable_like(contour, img_w: int, img_h: int) -> bool:
    """
    Return True if a contour looks like a cable/wire rather than a tool body.
    A cable is:
      • Very thin relative to its length  (minAreaRect aspect ratio > 5)
      • AND touches or exits the image border  (off-frame cable)
    """
    area = cv2.contourArea(contour)
    if area < 10:
        return False
    rect = cv2.minAreaRect(contour)
    short, long_ = sorted(rect[1])
    aspect = long_ / max(short, 1)
    if aspect < 4:
        return False          # not elongated enough
    # Check border touch
    x, y, bw, bh = cv2.boundingRect(contour)
    touches = (x <= 3 or y <= 3 or
               x + bw >= img_w - 3 or
               y + bh >= img_h - 3)
    return touches            # elongated AND off-frame → cable


def _score_contour(c, img_w: int, img_h: int, area_total: int) -> float:
    """
    Score a contour for likelihood of being the target object.
    Higher is better.

    Factors:
      + Closer to image centre (primary signal)
      - Very large (likely background bleed)
      - Cable-like shape touching border
      - Touches border (edge artifact)
      + Reasonable "object-like" size (0.3 % – 50 % of image)
    """
    c_area = cv2.contourArea(c)
    frac   = c_area / max(area_total, 1)

    # Hard reject: background bleed or specks
    if frac > 0.85 or frac < 0.001:
        return -1.0

    # Cable reject
    if _is_cable_like(c, img_w, img_h):
        return -1.0

    # Centre distance score (normalised so max image diagonal = 1)
    x, y, bw, bh = cv2.boundingRect(c)
    cx, cy   = x + bw / 2, y + bh / 2
    img_cx   = img_w / 2
    img_cy   = img_h / 2
    diag     = (img_w**2 + img_h**2) ** 0.5
    dist_n   = ((cx - img_cx)**2 + (cy - img_cy)**2) ** 0.5 / diag

    # Size score: prefer objects that fill 0.5 %–50 % of the image
    # bell-shaped peak at ~10 %
    size_score = 1.0 - abs(np.log10(max(frac, 1e-6)) + 1) / 3.0
    size_score = max(0.0, size_score)

    # Border penalty — contours touching an edge are suspect
    touches = (x <= 3 or y <= 3 or
               x + bw >= img_w - 3 or
               y + bh >= img_h - 3)
    border_penalty = 0.35 if touches else 0.0

    return size_score * (1.0 - dist_n) - border_penalty


def _threshold_bbox(img_bgr: np.ndarray):
    """
    Multi-method CV bounding-box detector  (v4 — all 9 sample images addressed).

    Methods run (A–F):
      A) Otsu ×2 polarities on greyscale
      B) CLAHE-enhanced greyscale Otsu ×2  (boosts low-contrast objects)
      C) LAB A-channel + B-channel ×4 variants  (coloured objects like orange marker)
      D) Bilateral-filter + adaptive threshold  (textured backgrounds: wood grain)
      E) CLAHE + Sobel-magnitude edges  (dark objects: buckle on wood)
      F) HSV value channel  (very dark tools on bright backgrounds)
      G) Canny + morphological fill  (sharp-edge objects)

    Contour selection:
      • Minimum area: 0.003 × image (was 0.015 — catches tiny buckle, pushpin)
      • Scores each contour on: closeness to centre, object-like size, border penalty,
        cable-shape penalty
      • Best-scoring candidate wins across ALL methods

    Post-processing:
      • Cable contours (thin + touching border) are filtered BEFORE scoring
      • Optional GrabCut refinement on the winner

    Returns ([x1,y1,x2,y2], binary_mask_uint8) or (None, None).
    """
    h, w = img_bgr.shape[:2]
    area_total = h * w
    candidates = []   # (score, bbox_list, mask)

    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Shared morphology kernels
    kc15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ko5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    k30  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    k8   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,  8))

    def _try_mask(mask_raw, big_kernel=False):
        """Morphologically clean mask, score all valid contours, add best."""
        kc = k30 if big_kernel else kc15
        ko = k8  if big_kernel else ko5
        m  = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kc)
        m  = cv2.morphologyEx(m,        cv2.MORPH_OPEN,  ko)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            sc = _score_contour(c, w, h, area_total)
            if sc <= 0:
                continue
            x_, y_, bw_, bh_ = cv2.boundingRect(c)
            candidates.append((sc, [x_, y_, x_ + bw_, y_ + bh_], m))

    # ── A) Otsu ×2 polarities ─────────────────────────────────────────────────
    for inv in (False, True):
        g = 255 - gray_blur if inv else gray_blur
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _try_mask(th)

    # ── B) CLAHE-enhanced Otsu ×2  (low-contrast boost) ──────────────────────
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)
    enh_b = cv2.GaussianBlur(enh, (5, 5), 0)
    for inv in (False, True):
        g = 255 - enh_b if inv else enh_b
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _try_mask(th)

    # ── C) LAB A and B channels ×4 variants  (coloured objects) ──────────────
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    for ch_i in (1, 2):
        ch = lab[:, :, ch_i]
        for flags in (cv2.THRESH_BINARY     + cv2.THRESH_OTSU,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU):
            _, th = cv2.threshold(ch, 0, 255, flags)
            _try_mask(th)

    # ── D) Bilateral + adaptive threshold  (textured bg: wood grain) ─────────
    bfilt = cv2.bilateralFilter(gray, 15, 80, 80)
    adap  = cv2.adaptiveThreshold(bfilt, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 5)
    _try_mask(adap)

    # ── E) CLAHE + Sobel magnitude  (edge-based, any-contrast) ───────────────
    sx  = cv2.Sobel(enh, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(enh, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.normalize(np.sqrt(sx**2 + sy**2), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    mag_b = cv2.GaussianBlur(mag, (9, 9), 0)
    _, sob_th = cv2.threshold(mag_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _try_mask(sob_th, big_kernel=True)

    # ── F) HSV value channel  (dark tool on bright background) ───────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    _, th_dark = cv2.threshold(val, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _try_mask(th_dark)

    # ── G) Canny + morphological fill  (sharp-edge objects) ──────────────────
    edges  = cv2.Canny(gray_blur, 30, 120)
    filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k30)
    _try_mask(filled, big_kernel=True)

    if not candidates:
        return None, None

    # Best candidate = highest score
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, bbox, binary_mask = candidates[0]

    # ── Optional GrabCut refinement ───────────────────────────────────────────
    try:
        x1, y1, x2, y2 = bbox
        bw2, bh2 = x2 - x1, y2 - y1
        if bw2 > 20 and bh2 > 20:
            rect = (max(1, x1), max(1, y1),
                    min(bw2, w - x1 - 1), min(bh2, h - y1 - 1))
            bgd_m = np.zeros((1, 65), np.float64)
            fgd_m = np.zeros((1, 65), np.float64)
            gc    = np.zeros(img_bgr.shape[:2], np.uint8)
            cv2.grabCut(img_bgr, gc, rect, bgd_m, fgd_m, 3, cv2.GC_INIT_WITH_RECT)
            gc_fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                              255, 0).astype(np.uint8)
            orig_area = (binary_mask > 0).sum()
            overlap   = np.logical_and(gc_fg > 0, binary_mask > 0).sum()
            if orig_area > 0 and overlap / orig_area > 0.35:
                binary_mask = gc_fg
    except Exception:
        pass

    pad = 12
    x1, y1, x2, y2 = bbox
    return ([max(0, x1 - pad), max(0, y1 - pad),
             min(w, x2 + pad), min(h, y2 + pad)],
            binary_mask)


# ─── YOLO model loader (cascade: YOLOv11x → v11m → v8x → v8m) ──────────────

_YOLO_MODEL_CASCADE = [
    ("yolo11x.pt",  "YOLOv11x"),   # Best accuracy — latest Ultralytics gen
    ("yolo11m.pt",  "YOLOv11m"),   # Lighter YOLOv11
    ("yolov8x.pt",  "YOLOv8x"),    # Fallback: proven YOLOv8 extra-large
    ("yolov8m.pt",  "YOLOv8m"),    # Smallest acceptable fallback
]

def _load_yolo() -> tuple:
    """
    Load the best available YOLO model from the cascade.
    Returns (model, model_name_str).
    Caches in _seg_models_cache["yolo_v3"].
    """
    if "yolo_v3" in _seg_models_cache:
        return _seg_models_cache["yolo_v3"]

    from ultralytics import YOLO

    for weights, name in _YOLO_MODEL_CASCADE:
        try:
            logger.info(f"Attempting to load {name} ({weights})…")
            model = YOLO(weights)
            _seg_models_cache["yolo_v3"] = (model, name)
            logger.info(f"✓ Loaded {name}")
            return model, name
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")

    raise RuntimeError("No YOLO model could be loaded from cascade")


# ─── Main detection function ─────────────────────────────────────────────────

def gradio_detect_tool(img_bgr: np.ndarray):
    """
    Robust single-object tool detector.

    Returns:
        bbox        [x1,y1,x2,y2]  — final merged bounding box for the tool
        thresh_mask np.ndarray      — CV binary mask (used by SAM as hint)
        method      str             — model name used
        warning     str | None      — non-None if multiple distinct objects found
        raw_boxes   list[list]      — individual merged-cluster boxes (for vis)
    """
    h, w = img_bgr.shape[:2]
    img_cx, img_cy = w / 2, h / 2

    # ── Always run CV detector (gives us a pixel mask regardless) ────────────
    thresh_bbox, thresh_mask = _threshold_bbox(img_bgr)

    warning     = None
    raw_boxes   = []   # merged cluster boxes returned to visualiser
    yolo_boxes  = []   # raw YOLO boxes before merging
    method      = "CV"

    # ── 1. YOLO detection ────────────────────────────────────────────────────
    try:
        model, model_name = _load_yolo()
        # Very low conf so we catch all parts of a multi-coloured tool
        result = model(img_bgr, conf=0.08, iou=0.30, verbose=False)[0]
        boxes  = result.boxes
        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if (x2 - x1) > 5 and (y2 - y1) > 5:   # skip degenerate boxes
                    yolo_boxes.append([x1, y1, x2, y2])
            method = model_name
            logger.info(f"{model_name}: {len(yolo_boxes)} raw boxes before merge")
        else:
            logger.info(f"{model_name}: no detections above threshold")
    except Exception as e:
        logger.warning(f"YOLO skipped: {e}")

    # ── 2. Merge fragments → whole-tool clusters ─────────────────────────────
    if yolo_boxes:
        merged_clusters = _merge_tool_boxes(yolo_boxes, w, h)
        raw_boxes = merged_clusters   # pass to visualiser

        if len(merged_clusters) == 1:
            final_bbox = merged_clusters[0]
            warning    = None
        else:
            # Multiple clusters — pick closest to image centre (most likely the tool)
            warning = ("⚠️  Multiple objects found! "
                       "Please get closer to the target tool.")
            logger.warning(f"Multiple clusters ({len(merged_clusters)}) — picking closest to centre")
            def _cluster_dist(box):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                return ((cx - img_cx)**2 + (cy - img_cy)**2) ** 0.5
            merged_clusters.sort(key=_cluster_dist)
            final_bbox = merged_clusters[0]

    else:
        # YOLO found nothing → fall back to CV
        if thresh_bbox is not None:
            final_bbox = thresh_bbox
            method     = "CV-Fallback"
            raw_boxes  = [thresh_bbox]
        else:
            pad_x = int(w * 0.15); pad_y = int(h * 0.15)
            final_bbox = [pad_x, pad_y, w - pad_x, h - pad_y]
            method     = "CentreCrop"
            raw_boxes  = [final_bbox]
            logger.warning("All detectors failed — using centre-crop")

    # ── Expand partial YOLO bbox using CV bbox ──────────────────────────────
    # YOLO sometimes only fires on the high-contrast part of a multi-color tool
    # (e.g. red grip of a screwdriver, black handles of scissors).
    # If the CV detector found a LARGER region, union the two boxes so SAM
    # gets the full tool extent — not just the colorful fragment.
    if thresh_bbox is not None:
        tx1, ty1, tx2, ty2 = thresh_bbox
        fx1, fy1, fx2, fy2 = final_bbox
        # Only expand — never shrink the YOLO box
        expanded = [
            min(fx1, tx1), min(fy1, ty1),
            max(fx2, tx2), max(fy2, ty2),
        ]
        # Accept expansion only if it's not the entire image (background bleed)
        exp_area  = (expanded[2]-expanded[0]) * (expanded[3]-expanded[1])
        img_area  = w * h
        if exp_area < 0.90 * img_area:
            logger.info(f"Bbox expanded by CV union: {final_bbox} → {expanded}")
            final_bbox = expanded
        else:
            logger.info("CV bbox expansion skipped (would cover >90% image)")

    # Add a breathing margin for SAM
    pad = 15
    x1, y1, x2, y2 = final_bbox
    final_bbox = [max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad)]

    logger.info(f"Final bbox ({method}): {final_bbox}  warning={warning is not None}")
    return final_bbox, thresh_mask, method, warning, raw_boxes


# ─── Visualisation ───────────────────────────────────────────────────────────

def gradio_draw_detection(img_bgr: np.ndarray,
                           final_bbox: list,
                           method: str,
                           warning: Optional[str],
                           raw_boxes: list) -> np.ndarray:
    """
    Draw detection result on a copy of img_bgr.
      • All merged cluster boxes shown in dim grey (context)
      • The selected tool box shown in bright green (or orange on warning)
      • Warning banner overlaid at top of image if multiple objects found
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Draw individual merged cluster boxes in dim colour
    for box in raw_boxes:
        bx1, by1, bx2, by2 = [int(v) for v in box]
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (180, 180, 60), 1)

    # Main selected box colour: orange = warning, green = ok
    main_colour = (0, 100, 255) if warning else (0, 230, 80)
    x1, y1, x2, y2 = [int(v) for v in final_bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), main_colour, 3)

    # Label tag
    label = f"TOOL  [{method}]"
    (tw, th_font), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    tag_y = max(y1 - 6, th_font + 4)
    cv2.rectangle(vis, (x1, tag_y - th_font - 4), (x1 + tw + 6, tag_y + 2),
                  main_colour, -1)
    cv2.putText(vis, label, (x1 + 3, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    # Warning banner across top of image
    if warning:
        banner_h = 46
        overlay  = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 40, 180), -1)
        cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)
        warn_text = "⚠  Multiple objects detected — move closer to your tool"
        cv2.putText(vis, warn_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 230, 60), 2, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITY: pick best segment from a segment-map using bbox overlap
# ─────────────────────────────────────────────────────────────────────────────

def _best_segment_by_bbox(seg_map: np.ndarray, segments: list,
                            bbox: list, img_h: int, img_w: int) -> np.ndarray:
    """
    Given a labelled segment map and a list of segment dicts, return a binary
    mask for the segment that best overlaps with bbox AND has a reasonable size.

    Scoring: overlap_ratio * size_score
        overlap_ratio = pixels of segment inside bbox / bbox area
        size_score    = 1  if  2% < segment_area < 70% of image
                      = 0  otherwise  (background or tiny noise)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    img_area  = img_h * img_w

    # Crop the seg_map to bbox region
    seg_crop = seg_map[max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)]

    best_score, best_mask = -1, None
    for seg in segments:
        sid  = seg["id"]
        full_mask  = (seg_map == sid)
        seg_area   = int(full_mask.sum())

        # Size filter: skip background (too large) or specks (too small)
        if seg_area < 0.01 * img_area or seg_area > 0.80 * img_area:
            continue

        # Overlap with bbox
        crop_mask    = (seg_crop == sid)
        overlap_px   = int(crop_mask.sum())
        overlap_ratio = overlap_px / bbox_area

        # Prefer segments that are a reasonable fraction of the image
        size_ratio = seg_area / img_area
        # Penalty for very large segments (likely background bleed)
        size_score = 1.0 - max(0.0, (size_ratio - 0.40) / 0.40)

        score = overlap_ratio * size_score
        logger.debug(f"  seg {sid}: area={seg_area} overlap={overlap_ratio:.2f} score={score:.3f}")

        if score > best_score:
            best_score = score
            best_mask  = full_mask.astype(np.uint8) * 255

    if best_mask is None:
        # All segments failed size/overlap filter — fall back to largest inside bbox
        logger.warning("No segment passed quality filter; using largest-in-bbox")
        best_count, best_mask = 0, None
        for seg in segments:
            sid  = seg["id"]
            crop_mask = (seg_crop == sid)
            cnt = int(crop_mask.sum())
            if cnt > best_count:
                best_count = cnt
                best_mask  = (seg_map == sid).astype(np.uint8) * 255
        if best_mask is None:
            best_mask = np.zeros((img_h, img_w), np.uint8)

    return best_mask


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_sam():
    if "sam" not in _seg_models_cache:
        from transformers import SamModel, SamProcessor
        import torch
        proc  = SamProcessor.from_pretrained("facebook/sam-vit-large") # base
        model = SamModel.from_pretrained("facebook/sam-vit-large")   # base
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device)
        _seg_models_cache["sam"] = (model, proc, device)
        logger.info(f"SAM loaded on {device}")
    return _seg_models_cache["sam"]


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION  (SAM only — GrabCut as fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _grabcut_seg(img_bgr: np.ndarray, bbox: list) -> np.ndarray:
    """Enhanced GrabCut — always used as fallback."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    # Shrink rect slightly inward so GrabCut doesn't mark edges as FG
    pad = max(2, min(bw, bh) // 10)
    rect = (max(0, x1 + pad), max(0, y1 + pad),
            max(1, bw - 2 * pad), max(1, bh - 2 * pad))
    try:
        mask_gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, mask_gc, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
        result = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
        # Morphological clean-up
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k)
        return result
    except Exception as e:
        logger.warning(f"GrabCut failed: {e}")
        out = np.zeros((h, w), np.uint8)
        out[y1:y2, x1:x2] = 255
        return out


def _find_shadow_neg_points(img_bgr: np.ndarray, mask: np.ndarray,
                             n_points: int = 3) -> list:
    """
    Identify candidate shadow pixels inside the mask and return their
    (x, y) coordinates as SAM negative-prompt points.

    Shadow properties (illumination physics):
      • Same chromaticity as background (cast shadow = background × scalar)
      • Darker than background in luminance
      • Lower gradient at the mask boundary (smooth transition, not a hard edge)
      • Found at the mask perimeter, not in the interior

    Returns a list of up to n_points (x, y) tuples, or [] if none found.
    """
    h, w = img_bgr.shape[:2]
    if np.count_nonzero(mask) < 200:
        return []

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # ── Sample background from image border strip ──────────────────────────
    bw = max(1, int(min(h, w) * 0.07))
    bdr = np.zeros((h, w), np.uint8)
    bdr[:bw, :] = 255; bdr[-bw:, :] = 255
    bdr[:, :bw] = 255; bdr[:, -bw:] = 255
    bg_px  = lab[bdr > 0]
    bg_L   = float(np.median(bg_px[:, 0]))
    bg_A   = float(np.median(bg_px[:, 1]))
    bg_B   = float(np.median(bg_px[:, 2]))
    bg_Lstd = max(float(np.std(bg_px[:, 0])), 1.0)
    bg_Astd = max(float(np.std(bg_px[:, 1])), 1.0)
    bg_Bstd = max(float(np.std(bg_px[:, 2])), 1.0)

    # ── Sample object core for L reference ────────────────────────────────
    ero_sz  = max(5, min(h, w) // 20)
    k_ero   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_sz, ero_sz))
    core    = cv2.erode(mask, k_ero)
    if np.count_nonzero(core) < 50:
        k_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core  = cv2.erode(mask, k_ero)
    obj_L = float(np.median(L[core > 0])) if np.count_nonzero(core) > 0 else bg_L * 0.4
    obj_bg_gap = abs(bg_L - obj_L)

    # ── Gradient magnitude (weak gradient → shadow-like boundary) ─────────
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx       = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sy       = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sx ** 2 + sy ** 2)
    grad_n   = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Boundary ring of the mask (RETR_EXTERNAL outer edge only) ─────────
    outer_c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_fill = np.zeros((h, w), np.uint8)
    cv2.drawContours(outer_fill, outer_c, -1, 255, -1)
    k_thin   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bnd_ring = cv2.bitwise_and(outer_fill,
                               cv2.bitwise_not(cv2.erode(outer_fill, k_thin, iterations=3)))

    # ── Shadow pixel criteria ──────────────────────────────────────────────
    # L must be between the object and the background (not as dark as real object,
    # not as bright as background).
    shadow_L_low  = obj_L + obj_bg_gap * 0.28   # 28% above object brightness
    shadow_L_high = bg_L  - bg_Lstd * 0.8       # 0.8 std below background
    if shadow_L_low >= shadow_L_high:            # degenerate case
        return []

    # Chromaticity must be close to background (shadow = background × scalar)
    chroma_thresh = max(10.0, (bg_Astd + bg_Bstd) * 8)

    # Gradient must be LOW at the boundary (shadow → smooth fade into background)
    bnd_grads = grad_n[bnd_ring > 0]
    if len(bnd_grads) == 0:
        return []
    grad_thresh = max(4.0, float(np.percentile(bnd_grads, 35)))

    shadow_seed_map = (
        (bnd_ring > 0) &
        (L > shadow_L_low)  & (L < shadow_L_high) &
        (np.abs(A - bg_A) < chroma_thresh) &
        (np.abs(B - bg_B) < chroma_thresh) &
        (grad_n < grad_thresh)
    ).astype(np.uint8) * 255

    if np.count_nonzero(shadow_seed_map) < 5:
        return []

    # ── Grow seeds into shadow-colored region ──────────────────────────────
    shadow_region = (
        (outer_fill > 0) &
        (L > shadow_L_low) & (L < shadow_L_high) &
        (np.abs(A - bg_A) < chroma_thresh * 1.5) &
        (np.abs(B - bg_B) < chroma_thresh * 1.5) &
        (grad_n < grad_thresh * 2.0)
    ).astype(np.uint8) * 255

    flooded = shadow_seed_map.copy()
    k_grow  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for _ in range(12):
        prev = np.count_nonzero(flooded)
        grown = cv2.dilate(flooded, k_grow)
        flooded = cv2.bitwise_and(grown, shadow_region)
        if np.count_nonzero(flooded) == prev:
            break

    # Safety: never take more than 30% of the mask area
    max_allowed = int(np.count_nonzero(mask) * 0.30)
    if np.count_nonzero(flooded) > max_allowed:
        logger.warning("Shadow region > 30% of mask — using seeds only")
        flooded = shadow_seed_map

    if np.count_nonzero(flooded) < 5:
        return []

    # ── Extract representative negative-prompt coordinates ────────────────
    # Use the centres of the largest connected components of the shadow region
    n_cc, lbl, sts, _ = cv2.connectedComponentsWithStats(flooded, 8)
    shadow_ccs = sorted(
        [(sts[i, cv2.CC_STAT_AREA],
          sts[i, cv2.CC_STAT_LEFT] + sts[i, cv2.CC_STAT_WIDTH]  // 2,
          sts[i, cv2.CC_STAT_TOP]  + sts[i, cv2.CC_STAT_HEIGHT] // 2)
         for i in range(1, n_cc)],
        reverse=True
    )
    neg_pts = [(int(cx), int(cy)) for _, cx, cy in shadow_ccs[:n_points]]
    logger.info(f"Shadow neg-prompt candidates: {neg_pts} "
                f"(shadow area={np.count_nonzero(flooded)}, "
                f"shadow_L=[{shadow_L_low:.1f},{shadow_L_high:.1f}])")
    return neg_pts


def _post_remove_shadow(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Post-processing fallback: remove shadow from mask using gradient-seeded
    flood fill when SAM negative-prompt re-run is not available.
    Uses the same shadow detection as _find_shadow_neg_points but directly
    erodes the mask instead of re-running SAM.
    """
    try:
        h, w = img_bgr.shape[:2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        bw = max(1, int(min(h, w) * 0.07))
        bdr = np.zeros((h, w), np.uint8)
        bdr[:bw, :] = 255; bdr[-bw:, :] = 255
        bdr[:, :bw] = 255; bdr[:, -bw:] = 255
        bg_px   = lab[bdr > 0]
        bg_L    = float(np.median(bg_px[:, 0]))
        bg_A    = float(np.median(bg_px[:, 1]))
        bg_B    = float(np.median(bg_px[:, 2]))
        bg_Lstd = max(float(np.std(bg_px[:, 0])), 1.0)
        bg_Astd = max(float(np.std(bg_px[:, 1])), 1.0)
        bg_Bstd = max(float(np.std(bg_px[:, 2])), 1.0)

        ero_sz = max(5, min(h, w) // 20)
        k_ero  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_sz, ero_sz))
        core   = cv2.erode(mask, k_ero)
        if np.count_nonzero(core) < 50:
            k_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            core  = cv2.erode(mask, k_ero)
        obj_L    = float(np.median(L[core > 0])) if np.count_nonzero(core) > 0 else bg_L * 0.4
        obj_bg_gap = abs(bg_L - obj_L)

        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sx, sy = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5), \
                 cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_n = cv2.normalize(np.sqrt(sx**2 + sy**2), None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)

        outer_c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_fill = np.zeros((h, w), np.uint8)
        cv2.drawContours(outer_fill, outer_c, -1, 255, -1)
        k3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bnd  = cv2.bitwise_and(outer_fill,
                               cv2.bitwise_not(cv2.erode(outer_fill, k3, iterations=3)))

        shadow_L_low  = obj_L + obj_bg_gap * 0.28
        shadow_L_high = bg_L  - bg_Lstd * 0.8
        if shadow_L_low >= shadow_L_high:
            return mask

        chroma_thresh = max(10.0, (bg_Astd + bg_Bstd) * 8)
        bnd_grads = grad_n[bnd > 0]
        if len(bnd_grads) == 0:
            return mask
        grad_thresh = max(4.0, float(np.percentile(bnd_grads, 35)))

        seeds = (
            (bnd > 0) &
            (L > shadow_L_low) & (L < shadow_L_high) &
            (np.abs(A - bg_A) < chroma_thresh) &
            (np.abs(B - bg_B) < chroma_thresh) &
            (grad_n < grad_thresh)
        ).astype(np.uint8) * 255

        if np.count_nonzero(seeds) < 5:
            return mask

        valid = (
            (outer_fill > 0) &
            (L > shadow_L_low) & (L < shadow_L_high) &
            (np.abs(A - bg_A) < chroma_thresh * 1.5) &
            (np.abs(B - bg_B) < chroma_thresh * 1.5) &
            (grad_n < grad_thresh * 2.0)
        ).astype(np.uint8) * 255

        flooded = seeds.copy()
        k_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for _ in range(12):
            p = np.count_nonzero(flooded)
            flooded = cv2.bitwise_and(cv2.dilate(flooded, k_g), valid)
            if np.count_nonzero(flooded) == p:
                break

        # Safety valve: max 25% removal
        max_px = int(np.count_nonzero(mask) * 0.25)
        if np.count_nonzero(flooded) > max_px:
            logger.warning("Post shadow removal > 25% — clamping to seeds")
            flooded = seeds

        if np.count_nonzero(flooded) < 5:
            return mask

        # Remove shadow from mask, keep largest CC, close holes
        k_dil  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        shadow_exp = cv2.bitwise_and(cv2.dilate(flooded, k_dil), mask)
        refined    = cv2.bitwise_and(mask, cv2.bitwise_not(shadow_exp))
        n2, l2, s2, _ = cv2.connectedComponentsWithStats(refined, 8)
        if n2 > 1:
            refined = (l2 == 1 + int(np.argmax(s2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
        k_cl  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k_cl)

        removal = 1.0 - np.count_nonzero(refined) / max(np.count_nonzero(mask), 1)
        logger.info(f"Post shadow removal: {removal*100:.1f}% removed")
        if removal > 0.35:
            logger.warning("Post shadow removal too aggressive — reverting")
            return mask
        return refined

    except Exception as e:
        logger.warning(f"_post_remove_shadow failed: {e}")
        return mask


def seg_sam(img_bgr: np.ndarray, bbox: list) -> np.ndarray:
    """
    SAM segmentation with two-pass shadow removal.

    The bbox passed in already covers the FULL tool (all YOLO boxes unioned),
    so a single well-placed center-point prompt is sufficient.
    Grid prompts are NOT used — for tools with holes (scissors finger rings,
    wrenches) evenly-spaced grid points land on background inside the holes,
    confusing SAM and producing partial masks.

    Pass 1 — SAM run: bbox (full tool extent) + single centre-point prompt.
              SAM's bbox conditioning alone is usually enough to segment the
              whole tool; the center point disambiguates foreground vs background.
    Pass 2 — Shadow detection: re-run with shadow pixels as NEGATIVE prompts
              so SAM learns to exclude the cast shadow from the mask.
    Fallback — Post-process shadow removal on the mask if pass-2 fails.
    """
    h, w = img_bgr.shape[:2]
    try:
        import torch
        model, proc, device = _load_sam()
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # ── _run_sam helper ───────────────────────────────────────────────────
        def _run_sam(pos_pts, neg_pts):
            all_pts    = pos_pts + neg_pts
            all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)
            inp = proc(
                img_pil,
                input_boxes  = [[[x1, y1, x2, y2]]],
                input_points = [all_pts],
                input_labels = [all_labels],
                return_tensors = "pt"
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            masks_out = proc.image_processor.post_process_masks(
                out.pred_masks.cpu(),
                inp["original_sizes"].cpu(),
                inp["reshaped_input_sizes"].cpu()
            )
            scores = out.iou_scores[0, 0].cpu().numpy()
            best_i = int(scores.argmax())
            m = masks_out[0][0][best_i].numpy().astype(np.uint8)
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        # ── Pass 1: initial mask ─────────────────────────────────────────────
        # Use center + 4 bbox-edge midpoints as positive prompts.
        # Edge midpoints reach the blade tip and handle extremes without
        # landing inside finger holes (which are in the interior of the bbox).
        bx_mid = (x1 + x2) // 2
        by_mid = (y1 + y2) // 2
        # Inset slightly from the edge so we're clearly inside the tool
        inset_x = max(5, (x2 - x1) // 8)
        inset_y = max(5, (y2 - y1) // 8)
        edge_pts = [
            [bx_mid,        y1 + inset_y],   # top-mid   → blade tip
            [bx_mid,        y2 - inset_y],   # bottom-mid → handle base
            [x1 + inset_x,  by_mid       ],   # left-mid
            [x2 - inset_x,  by_mid       ],   # right-mid
        ]
        pos_pts = [[cx, cy]] + edge_pts
        logger.info(f"SAM pass-1: {len(pos_pts)} positive prompts (center + 4 edge-mids)")

        mask_np = _run_sam(pos_pts, [])
        result  = (mask_np > 0).astype(np.uint8) * 255

        # Sanity: near-full mask = SAM grabbed background → invert
        if np.count_nonzero(result) > 0.80 * h * w:
            logger.warning("SAM pass-1: near-full mask → inverting")
            inv = np.zeros((h, w), np.uint8)
            inv[y1:y2, x1:x2] = 255
            inv = cv2.bitwise_and(inv, cv2.bitwise_not(result))
            result = inv if np.count_nonzero(inv) > 100 else result

        bbox_area = max((x2 - x1) * (y2 - y1), 1)
        coverage  = np.count_nonzero(result[y1:y2, x1:x2]) / bbox_area
        logger.info(f"SAM pass-1 bbox coverage: {coverage*100:.1f}%")

        # ── Pass 2: shadow-aware re-run ───────────────────────────────────────
        neg_pts = _find_shadow_neg_points(img_bgr, result, n_points=3)
        if neg_pts:
            logger.info(f"SAM pass-2: re-running with {len(neg_pts)} shadow neg-prompts")
            try:
                mask_np2 = _run_sam(pos_pts, neg_pts)
                result2  = (mask_np2 > 0).astype(np.uint8) * 255
                area1, area2 = np.count_nonzero(result), np.count_nonzero(result2)
                shrink = 1.0 - area2 / max(area1, 1)
                if 0.0 < shrink < 0.40:
                    logger.info(f"SAM pass-2 accepted: {shrink*100:.1f}% shadow removed")
                    result = result2
                else:
                    logger.warning(f"SAM pass-2 rejected (shrink={shrink*100:.1f}%) — "
                                   "falling back to post-process")
                    result = _post_remove_shadow(img_bgr, result)
            except Exception as e2:
                logger.warning(f"SAM pass-2 failed ({e2}) — applying post-processing")
                result = _post_remove_shadow(img_bgr, result)
        else:
            result = _post_remove_shadow(img_bgr, result)

        return result

    except Exception as e:
        logger.warning(f"SAM failed: {e}")
        return _grabcut_seg(img_bgr, bbox)



# ─────────────────────────────────────────────────────────────────────────────
#  CORE PIPELINE  mask → contour → DXF

def _render_dxf_preview_img(dxf_pts: list, model_label: str,
                              W=600, H=520) -> np.ndarray:
    """Render DXF contour points as a red-lines-on-white preview image."""
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    if not dxf_pts or len(dxf_pts) < 2:
        cv2.putText(canvas, "No DXF points", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        return canvas

    pts  = np.array(dxf_pts, dtype=np.float32)
    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    rng  = max(xmax - xmin, ymax - ymin, 1e-3)
    PAD  = 40
    sc   = (min(W, H) - 2 * PAD) / rng

    def to_px(x, y):
        return (int(PAD + (x - xmin) * sc),
                int(H - PAD - (y - ymin) * sc))

    closed = dxf_pts + [dxf_pts[0]]
    for i in range(len(closed) - 1):
        p1 = to_px(*closed[i])
        p2 = to_px(*closed[i + 1])
        cv2.line(canvas, p1, p2, (0, 0, 210), 2, cv2.LINE_AA)

    cv2.putText(canvas, model_label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def run_single_model_pipeline(img_bgr: np.ndarray,
                               binary_mask: np.ndarray,
                               length_inches: float,
                               depth_inches: float,
                               model_name: str) -> dict:
    """
    binary_mask → contour → smoothing → DXF.
    Mirrors the exact logic of api_image_to_dxf in the original code.
    """
    result = {k: None for k in
              ("mask_vis", "contour_img", "merged_img", "dxf_preview", "dxf_path")}

    try:
        img_h, img_w = img_bgr.shape[:2]

        # Resize mask if needed
        if binary_mask.shape[:2] != (img_h, img_w):
            binary_mask = cv2.resize(binary_mask, (img_w, img_h),
                                     interpolation=cv2.INTER_NEAREST)

        # Binarise
        if len(binary_mask.shape) == 3:
            mg = (cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
                  if binary_mask.shape[2] == 3 else binary_mask[:, :, 3])
        else:
            mg = binary_mask
        _, mask = cv2.threshold(mg, 127, 255, cv2.THRESH_BINARY)

        if np.count_nonzero(mask) < 50:
            logger.warning(f"{model_name}: mask essentially empty — skip")
            return result

        # ── Mask visualisation ─────────────────────────────────────────────────
        mask_vis = img_bgr.copy()
        overlay  = mask_vis.copy()
        overlay[mask > 0] = (0, 200, 100)
        cv2.addWeighted(overlay, 0.45, mask_vis, 0.55, 0, mask_vis)
        result["mask_vis"] = mask_vis

        # ── Find largest contour ───────────────────────────────────────────────
        PAD = 1
        mask_padded = cv2.copyMakeBorder(mask, PAD, PAD, PAD, PAD,
                                          cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(mask_padded, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        if not contours:
            logger.warning(f"{model_name}: no contours in mask")
            return result
        main_contour = max(contours, key=cv2.contourArea) - PAD
        logger.info(f"{model_name}: raw contour {len(main_contour)} pts")

        # ── Spike removal + geometry-preserving SG smoothing ─────────────────
        # Step 1: remove outlier spike points BEFORE SG so they don't
        #         corrupt the polynomial fit inside the SG window.
        _sp = main_contour.reshape(-1, 2).astype(np.float64)
        _sp = remove_contour_spikes(_sp, window=5, z_thresh=3.5)
        main_contour = _sp.reshape(-1, 1, 2).astype(np.int32)

        # Step 2: two-pass SG (tight → wider) to flatten staircase and waviness.
        for _window in (11, 21):
            main_contour = micro_smooth_contour_pixels(
                main_contour,
                corner_angle_threshold=20.0,
                straight_angle_threshold=3.0,
                corner_guard=2,
            )

        # ── Tool offset (same as original) ────────────────────────────────────
        tool_offset_inches = get_tool_offset_inches()
        temp_scale         = img_h / max(length_inches, 1e-6)
        off_px             = tool_offset_inches * temp_scale
        pts_off = [(p[0][0], p[0][1]) for p in main_contour]
        try:
            from shapely.geometry import Polygon as _Poly
            poly_off = _Poly(pts_off)
            if not poly_off.is_valid:
                poly_off = poly_off.buffer(0)
            poly_off = poly_off.buffer(off_px)
            if poly_off.geom_type == 'MultiPolygon':
                poly_off = max(poly_off.geoms, key=lambda p: p.area)
            main_contour = (np.array(poly_off.exterior.coords, dtype=np.float32)
                            .reshape(-1, 1, 2).astype(np.int32))
        except Exception as e:
            logger.warning(f"{model_name}: offset polygon failed — {e}")

        # Contour thickness (pixels)
        ct_px = max(2, int((0.05 / max(length_inches, 1e-6)) * img_h))

        # ── Shapely expansion for foam pocket ─────────────────────────────────
        exp_px  = (0.25 / max(length_inches, 1e-6)) * img_h
        pts_exp = [(p[0][0], p[0][1]) for p in main_contour]
        try:
            from shapely.geometry import Polygon as _Poly
            poly = _Poly(pts_exp)
            if not poly.is_valid:
                poly = poly.buffer(0)
            exp_poly = poly.buffer(exp_px)
            if exp_poly.geom_type == 'MultiPolygon':
                exp_poly = max(exp_poly.geoms, key=lambda p: p.area)
            main_c_smooth = (np.array(poly.exterior.coords, dtype=np.float32)
                             .reshape(-1, 1, 2).astype(np.int32))
            exp_c_smooth  = (np.array(exp_poly.exterior.coords, dtype=np.float32)
                             .reshape(-1, 1, 2).astype(np.int32))
        except Exception as e:
            logger.warning(f"{model_name}: Shapely expansion failed — {e}")
            main_c_smooth = main_contour
            exp_c_smooth  = main_contour

        # ── Canvas expansion to fit both contours ─────────────────────────────
        x_m, y_m, w_m, h_m = cv2.boundingRect(main_c_smooth)
        x_e, y_e, w_e, h_e = cv2.boundingRect(exp_c_smooth)
        PAD2  = 1
        x_min = min(x_m, x_e);          y_min = min(y_m, y_e)
        x_max = max(x_m + w_m, x_e + w_e); y_max = max(y_m + h_m, y_e + h_e)
        off_x  = max(0, PAD2 - x_min);  off_y = max(0, PAD2 - y_min)
        new_w  = img_w + off_x + max(0, (x_max + PAD2) - img_w)
        new_h  = img_h + off_y + max(0, (y_max + PAD2) - img_h)

        main_c_off = main_c_smooth + [off_x, off_y]
        exp_c_off  = exp_c_smooth  + [off_x, off_y]

        # ── DXF generation ────────────────────────────────────────────────────
        scale_f = ((length_inches + 2 * tool_offset_inches) * 25.4) / max(h_m, 1)
        dxf_pts = []
        for pt in main_c_smooth:
            xp, yp = pt[0]
            dxf_pts.append(((xp - x_m) * scale_f,
                             (h_m - (yp - y_m)) * scale_f))
            #------------------------------------------------------------------------------
        # dxf_pts = remove_dxf_spikes(dxf_pts, window=5, z_thresh=3.5)
        # dxf_pts = smooth_dxf_points_mm(dxf_pts)

        # doc = ezdxf.new(units=ezdxf.units.MM)
        # doc.header["$INSUNITS"] = ezdxf.units.MM
        # msp = doc.modelspace()
        # # Write as lwpolyline through all smoothed points so DXF matches preview exactly.
        # msp.add_lwpolyline(dxf_pts, close=True,
        #                    dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})

        dxf_pts = remove_dxf_spikes(dxf_pts, window=5, z_thresh=3.5)
        dxf_pts = smooth_dxf_points_mm(dxf_pts)
        dxf_pts = remove_dxf_spikes(dxf_pts, window=3, z_thresh=2.0)

        doc = ezdxf.new(units=ezdxf.units.MM)
        doc.header["$INSUNITS"] = ezdxf.units.MM
        msp = doc.modelspace()
        # Write as cubic spline with point reduction — eliminates micro-spikes
        # by letting the CAD viewer interpolate smooth curves between fit points.
        # Every 6th point keeps geometry accurate while removing zigzag artifacts.
        reduced_pts = dxf_pts[::6]
        if len(reduced_pts) < 4:
            reduced_pts = dxf_pts[::2]
        if len(reduced_pts) < 4:
            reduced_pts = dxf_pts
        spline_pts = reduced_pts + [reduced_pts[0]]  # close the spline
        msp.add_spline(fit_points=spline_pts, degree=3,
                       dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})

            #-----------------------------------------------------------------------------
        



        
        # meta_x = max((p[0] for p in dxf_pts), default=0) + 5
        # for txt, dy in [(f"Model: {model_name}", 0),
        #                 (f"Length: {length_inches:.3f} in", -8),
        #                 (f"Depth:  {depth_inches:.3f} in", -16)]:
        #     msp.add_text(txt, dxfattribs={
        #         "layer": "META", "color": 7, "height": 3
        #     }).set_placement((meta_x, dy))

        uid       = uuid.uuid4().hex[:8]
        safe_nm   = model_name.replace(" ", "_").replace("/", "_")
        dxf_path  = os.path.join(GRADIO_OUTPUT_DIR, f"{safe_nm}_{uid}.dxf")
        doc.saveas(dxf_path)
        result["dxf_path"]    = dxf_path
        result["dxf_preview"] = _render_dxf_preview_img(dxf_pts, model_name)

        # ── Merged image (gold tight + grey expanded) ──────────────────────────
        if img_bgr.shape[2] == 3:
            img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = 255
        else:
            img_rgba = img_bgr.copy()

        merged = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        merged[off_y:off_y + img_h, off_x:off_x + img_w] = img_rgba

        gray_bg = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        cv2.fillPoly(gray_bg, [exp_c_off], (194, 194, 194, 255))

        a_tool = merged[:, :, 3] / 255.0
        a_gray = gray_bg[:, :, 3] / 255.0 * (1.0 - a_tool)
        for ch in range(3):
            merged[:, :, ch] = (a_tool * merged[:, :, ch] +
                                 a_gray * gray_bg[:, :, ch])
        merged[:, :, 3] = np.maximum(merged[:, :, 3], gray_bg[:, :, 3])

        cv2.polylines(merged, [main_c_off], True, (168, 108, 38, 255), ct_px, cv2.LINE_AA)
        cv2.polylines(merged, [exp_c_off],  True, (100, 100, 100, 255), 1,    cv2.LINE_AA)
        result["merged_img"] = cv2.cvtColor(merged, cv2.COLOR_BGRA2BGR)

        # ── Contour on original image ──────────────────────────────────────────
        contour_img = img_bgr.copy()
        cv2.drawContours(contour_img, [main_c_smooth], -1,
                         (255, 80, 0), ct_px, cv2.LINE_AA)
        result["contour_img"] = contour_img

        logger.info(f"{model_name}: pipeline complete ✓")

    except Exception as e:
        logger.error(f"{model_name} pipeline error: {e}", exc_info=True)

    return result



# ═══════════════════════════════════════════════════════════════════════════════
#  LAUNCH  —  Flask only, no Gradio
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    logger.info(f"Starting Flask API (no Gradio) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
