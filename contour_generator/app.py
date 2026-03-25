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
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            entities = list(msp)
            
            if not entities:
                logger.error(f"DXF file has no entities: {dxf_path}")
                return 0.0, 0.0, 0.0, []
            
            logger.info(f"DXF has {len(entities)} entities, types: {[type(e).__name__ for e in entities[:5]]}")
            
            # CRITICAL: Convert Spline entities to LWPolyline for processing
            converted_entities = []
            for entity in entities:
                entity_type = type(entity).__name__
                if entity_type == 'Spline' or entity_type == 'SPLINE':
                    try:
                        logger.info(f"Converting Spline to LWPolyline...")
                        polyline_points = []
                        
                        # Method 1: Try flattening with error handling
                        try:
                            flat_points = list(entity.flattening(distance=0.1))
                            if flat_points and len(flat_points) >= 2:
                                for pt in flat_points:
                                    if hasattr(pt, 'x'):
                                        polyline_points.append((pt.x, pt.y))
                                    elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                                        polyline_points.append((pt[0], pt[1]))
                                logger.info(f"Extracted {len(polyline_points)} points via flattening")
                        except Exception as flatten_ex:
                            logger.warning(f"Flattening failed: {flatten_ex}, trying direct point access...")
                        
                        # Method 2: Try fit_points if flattening failed
                        if not polyline_points:
                            try:
                                fit_pts = entity.fit_points
                                if fit_pts:
                                    for pt in fit_pts:
                                        if hasattr(pt, 'x'):
                                            polyline_points.append((pt.x, pt.y))
                                        elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                                            polyline_points.append((pt[0], pt[1]))
                                    logger.info(f"Extracted {len(polyline_points)} fit points")
                            except Exception as fit_ex:
                                logger.warning(f"Fit points failed: {fit_ex}")
                        
                        # Method 3: Try control_points
                        if not polyline_points:
                            try:
                                ctrl_pts = entity.control_points
                                if ctrl_pts:
                                    for pt in ctrl_pts:
                                        if hasattr(pt, 'x'):
                                            polyline_points.append((pt.x, pt.y))
                                        elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                                            polyline_points.append((pt[0], pt[1]))
                                    logger.info(f"Extracted {len(polyline_points)} control points")
                            except Exception as ctrl_ex:
                                logger.warning(f"Control points failed: {ctrl_ex}")
                        
                        # Method 4: Approximate by sampling the spline at parameter values
                        if not polyline_points:
                            try:
                                logger.info(f"Trying parameter-based approximation...")
                                # Get knot values to determine valid parameter range
                                try:
                                    # For NURBS splines, parameters go from knots[0] to knots[-1]
                                    knots = entity.knots
                                    if knots and len(knots) >= 2:
                                        t_start = knots[0]
                                        t_end = knots[-1]
                                    else:
                                        # Default to 0-1 range
                                        t_start = 0.0
                                        t_end = 1.0
                                except:
                                    # Fallback to 0-1 range
                                    t_start = 0.0
                                    t_end = 1.0
                                
                                # Sample at regular intervals
                                try:
                                    n_ctrl = entity.control_point_count() if hasattr(entity, 'control_point_count') else 10
                                except:
                                    n_ctrl = 10
                                samples = max(20, n_ctrl * 5)
                                
                                for i in range(samples + 1):
                                    t = t_start + (t_end - t_start) * (i / samples)
                                    try:
                                        # Try different methods to evaluate the spline
                                        point = None
                                        
                                        # Method A: Try to get point at parameter t
                                        if hasattr(entity, 'point_at'):
                                            point = entity.point_at(t)
                                        # Method B: Try construction class method
                                        elif hasattr(entity, 'construction_tool'):
                                            try:
                                                spline_tool = entity.construction_tool()
                                                if hasattr(spline_tool, 'point'):
                                                    point = spline_tool.point(t)
                                            except:
                                                pass
                                        
                                        if point:
                                            if hasattr(point, 'x'):
                                                polyline_points.append((point.x, point.y))
                                            elif isinstance(point, (tuple, list)) and len(point) >= 2:
                                                polyline_points.append((point[0], point[1]))
                                    except Exception as eval_ex:
                                        logger.debug(f"Failed to evaluate at t={t}: {eval_ex}")
                                        pass
                                
                                if polyline_points:
                                    logger.info(f"Approximated spline with {len(polyline_points)} sampled points")
                                else:
                                    logger.warning(f"Parameter sampling produced no points")
                            except Exception as approx_ex:
                                logger.warning(f"Parameter approximation failed: {approx_ex}")
                        
                        # === CRITICAL FIX: Create STANDALONE LWPolyline ===
                        # DO NOT add to msp yet - create a standalone entity
                        if polyline_points and len(polyline_points) >= 2:
                            try:
                                # Create a TEMPORARY document just to create the entity
                                temp_doc = ezdxf.new()
                                temp_msp = temp_doc.modelspace()
                                
                                is_closed = getattr(entity, 'closed', False)
                                # Add to TEMP msp, then copy the entity
                                temp_polyline = temp_msp.add_lwpolyline(polyline_points, close=is_closed)
                                
                                # Copy the entity to get a standalone version
                                standalone_polyline = temp_polyline.copy()
                                
                                converted_entities.append(standalone_polyline)
                                logger.info(f"Successfully converted Spline to LWPolyline with {len(polyline_points)} points")
                            except Exception as create_ex:
                                logger.error(f"Failed to create LWPolyline: {create_ex}, keeping original")
                                converted_entities.append(entity)
                        else:
                            logger.error(f"All point extraction methods failed, keeping original Spline")
                            converted_entities.append(entity)
                            
                    except Exception as spline_conv_ex:
                        logger.error(f"Spline conversion failed: {spline_conv_ex}")
                        converted_entities.append(entity)
                else:
                    converted_entities.append(entity)
            
            # Use converted entities for processing
            entities = converted_entities
            logger.info(f"After conversion: {len(entities)} entities, types: {[type(e).__name__ for e in entities[:5]]}")
            
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            points_extracted = 0
            for entity in entities:
                try:
                    entity_type = type(entity).__name__
                    points = self._extract_entity_points(entity)
                    if points:
                        points_extracted += len(points)
                        logger.debug(f"Extracted {len(points)} points from {entity_type}")
                        for point in points:
                            try:
                                # Try to access .x attribute (Vec3, Vec2)
                                if hasattr(point, 'x'):
                                    try:
                                        x, y, z = point.x, point.y, getattr(point, 'z', 0)
                                    except Exception as attr_ex:
                                        if points_extracted < 3:
                                            logger.error(f"Point HAS .x but accessing it failed: {attr_ex}, type={type(point).__name__}")
                                        continue
                                # Try tuple/list indexing (including numpy arrays)
                                elif hasattr(point, '__getitem__') and hasattr(point, '__len__') and len(point) >= 2:
                                    x, y = float(point[0]), float(point[1])
                                    z = float(point[2]) if len(point) > 2 else 0.0
                                else:
                                    if points_extracted < 3:
                                        logger.warning(f"Point format unrecognized: type={type(point).__name__}, has_x={hasattr(point, 'x')}, has_getitem={hasattr(point, '__getitem__')}, repr={repr(point)[:100]}")
                                    continue
                            except Exception as point_ex:
                                if points_extracted < 3:
                                    logger.error(f"Point processing exception: {point_ex}, type={type(point).__name__}")
                                continue
                            min_x = min(min_x, x)
                            max_x = max(max_x, x)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y)
                            min_z = min(min_z, z)
                            max_z = max(max_z, z)
                    else:
                        logger.warning(f"No points from entity type: {entity_type}")
                except Exception as ex:
                    logger.warning(f"Failed processing entity {type(entity).__name__}: {ex}")
                    continue
            
            logger.info(f"Total points extracted: {points_extracted}")
            
            # CRITICAL: Check if we got any valid points
            if min_x == float('inf'):
                logger.error(f"No valid points extracted from DXF: {dxf_path}")
                logger.error(f"Entity types present: {[type(e).__name__ for e in entities]}")
                return 0.0, 0.0, 0.0, []
            
            width_mm = max_x - min_x
            height_mm = max_y - min_y
            depth_mm = max(max_z - min_z, 1.0)
            
            logger.info(f"DXF dimensions calculated: {width_mm:.2f}mm x {height_mm:.2f}mm x {depth_mm:.2f}mm")
            
            normalized_entities = []
            for entity in entities:
                try:
                    normalized_entity = entity.copy()
                    self._translate_entity_manually(normalized_entity, -min_x, -min_y, -min_z)
                    normalized_entities.append(normalized_entity)
                except Exception as ex:
                    logger.warning(f"Failed normalizing entity: {ex}")
                    try:
                        normalized_entities.append(entity.copy())
                    except:
                        pass
            
            logger.info(f"Normalized {len(normalized_entities)} entities")
            
            return width_mm, height_mm, depth_mm, normalized_entities
            
        except Exception as e:
            logger.error(f"DXF analysis failed for {dxf_path}: {e}", exc_info=True)
            return 0.0, 0.0, 0.0, []
            
    def _scale_entities_to_target_size(self, entities: List, scale_factor: float) -> List:
        """Scale all entities by a uniform factor"""
        if abs(scale_factor - 1.0) < 0.001:
            return entities
            
        scaled_entities = []
        
        for entity in entities:
            try:
                scaled_entity = entity.copy()
                entity_type = type(scaled_entity).__name__
                
                # LWPOLYLINE - most common type
                if entity_type == 'LWPolyline':
                    # Get current vertices as list
                    current_vertices = list(scaled_entity.vertices())
                    # Clear and rebuild with scaled coordinates
                    scaled_entity.clear()
                    for vertex in current_vertices:
                        if len(vertex) >= 2:
                            scaled_entity.append((vertex[0] * scale_factor, vertex[1] * scale_factor))
                        
                # LINE
                elif hasattr(scaled_entity.dxf, 'start') and hasattr(scaled_entity.dxf, 'end'):
                    start = scaled_entity.dxf.start
                    end = scaled_entity.dxf.end
                    scaled_entity.dxf.start = (start.x * scale_factor, start.y * scale_factor, start.z)
                    scaled_entity.dxf.end = (end.x * scale_factor, end.y * scale_factor, end.z)
                        
                # CIRCLE or ARC
                elif hasattr(scaled_entity.dxf, 'center') and hasattr(scaled_entity.dxf, 'radius'):
                    center = scaled_entity.dxf.center
                    scaled_entity.dxf.center = (center.x * scale_factor, center.y * scale_factor, center.z)
                    scaled_entity.dxf.radius *= scale_factor
                
                scaled_entities.append(scaled_entity)
                
            except Exception as e:
                logger.error(f"Scale failed for {type(entity).__name__}: {e}")
                scaled_entities.append(entity)  # Use original if scaling fails
        
        return scaled_entities
            
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
        """
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
                    
                    # === STEP 1: Match expected dimensions ===
                    if tool.height_diagonal_inches > 0:
                        expected_height_mm = self._inches_to_mm(tool.height_diagonal_inches)
                        
                        # CRITICAL: Validate height_mm before division
                        if height_mm == 0:
                            logger.error(f"Tool '{tool.name}' has zero height - cannot scale")
                            raise Exception(f"Invalid tool dimensions for '{tool.name}': height is zero")
                        
                        if abs(height_mm - expected_height_mm) > 2.0:  # raised from 1.0 — 0.02in offset = 1.016mm, must not fire
                            scale_factor = expected_height_mm / height_mm
                            logger.info(f"Tool '{tool.name}' scaling to expected size: factor {scale_factor:.4f}")
                            normalized_entities = self._scale_entities_to_target_size(normalized_entities, scale_factor)
                            width_mm = width_mm * scale_factor
                            height_mm = expected_height_mm
                    
                    # === STEP 2: Apply 0.81in reduction (your existing logic) ===
                    reduction_inches = 0.81
                    current_height_inches = self._mm_to_inches(height_mm)
                    new_height_inches = current_height_inches - reduction_inches
                    
                    if new_height_inches > 0:
                        reduction_scale_factor = new_height_inches / current_height_inches
                        logger.info(f"Applying 0.81in reduction to '{tool.name}': "
                                   f"{current_height_inches:.3f}in -> {new_height_inches:.3f}in")
                        normalized_entities = self._scale_entities_to_target_size(normalized_entities, reduction_scale_factor)
                        width_mm = width_mm * reduction_scale_factor
                        height_mm = height_mm * reduction_scale_factor
                    
                    # === STEP 3: NEW - Apply 1-inch EXPANSION around center ===
                    # expansion_inches = 1.0
                    # current_height_inches = self._mm_to_inches(height_mm)
                    # current_width_inches = self._mm_to_inches(width_mm)
                    
                    # # STORE ORIGINAL DIMENSIONS BEFORE EXPANSION
                    # tool.original_width_mm = width_mm
                    # tool.original_height_mm = height_mm
                    
                    # target_height_inches = current_height_inches + expansion_inches
                    # expansion_scale_factor = target_height_inches / current_height_inches
                    
                    # logger.info(f"Applying 1in expansion to '{tool.name}': "
                    #            f"{current_height_inches:.3f}in -> {target_height_inches:.3f}in "
                    #            f"(scale factor: {expansion_scale_factor:.4f})")
                    
                    # # Calculate tool center BEFORE scaling
                    # tool_center_x_mm = width_mm / 2
                    # tool_center_y_mm = height_mm / 2
                    
                    # # Translate entities to origin (center at 0,0)
                    # for entity in normalized_entities:
                    #     self._translate_entity_manually(entity, -tool_center_x_mm, -tool_center_y_mm, 0)
                    
                    # # Scale around origin (which is now the center)
                    # normalized_entities = self._scale_entities_to_target_size(normalized_entities, expansion_scale_factor)
                    
                    # # Update dimensions
                    # width_mm = width_mm * expansion_scale_factor
                    # height_mm = height_mm * expansion_scale_factor
                    
                    # # Translate back so bottom-left is at origin
                    # new_center_x_mm = width_mm / 2
                    # new_center_y_mm = height_mm / 2
                    # for entity in normalized_entities:
                    #     self._translate_entity_manually(entity, new_center_x_mm, new_center_y_mm, 0)
                    
                    # logger.info(f"Tool '{tool.name}': Final dimensions = {width_mm:.2f}mm x {height_mm:.2f}mm")

                    # === STEP 3: NEW - Apply 1-inch EXPANSION around center ===
                    # === STEP 3: NEW - Apply 1-inch EXPANSION around center ===
                    expansion_inches = 0.84  # equals reduction_inches so Step2+Step3 net = 0
                    current_height_inches = self._mm_to_inches(height_mm)
                    current_width_inches = self._mm_to_inches(width_mm)
                    
                    # STORE ORIGINAL DIMENSIONS BEFORE EXPANSION
                    tool.original_width_mm = width_mm
                    tool.original_height_mm = height_mm
                    
                    # Calculate expansion scale factor
                    target_height_inches = current_height_inches + expansion_inches
                    expansion_scale_factor = target_height_inches / current_height_inches
                    
                    logger.info(f"Applying 1in expansion to '{tool.name}': "
                               f"{current_height_inches:.3f}in -> {target_height_inches:.3f}in "
                               f"(scale factor: {expansion_scale_factor:.4f})")
                    
                    # METHOD 1: Calculate ACTUAL bounding box of all entities
                    min_x = min_y = float('inf')
                    max_x = max_y = float('-inf')
                    
                    for entity in normalized_entities:
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
                    
                    # If we couldn't calculate bounds, use geometric center
                    if min_x == float('inf'):
                        center_x = width_mm / 2
                        center_y = height_mm / 2
                    else:
                        # Calculate center of actual bounding box
                        center_x = (min_x + max_x) / 2
                        center_y = (min_y + max_y) / 2
                        
                        logger.debug(f"Actual bounds: ({min_x:.2f},{min_y:.2f}) to ({max_x:.2f},{max_y:.2f})")
                        logger.debug(f"Geometric center: ({width_mm/2:.2f},{height_mm/2:.2f}) vs Actual center: ({center_x:.2f},{center_y:.2f})")
                    
                    # Store the actual center for position compensation later
                    tool.actual_center_x_mm = center_x
                    tool.actual_center_y_mm = center_y
                    
                    # Translate entities so actual center is at origin
                    for entity in normalized_entities:
                        self._translate_entity_manually(entity, -center_x, -center_y, 0)
                    
                    # Scale around origin (which is now the actual center)
                    normalized_entities = self._scale_entities_to_target_size(normalized_entities, expansion_scale_factor)
                    
                    # Update dimensions
                    width_mm = width_mm * expansion_scale_factor
                    height_mm = height_mm * expansion_scale_factor
                    
                    # Translate back so actual center returns to original position
                    for entity in normalized_entities:
                        self._translate_entity_manually(entity, center_x, center_y, 0)
                    
                    logger.info(f"Tool '{tool.name}': Final dimensions = {width_mm:.2f}mm x {height_mm:.2f}mm, "
                               f"Expanded around actual center ({center_x:.2f}, {center_y:.2f})")
                    
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
                for entity in tool.entities:
                    try:
                        new_entity = entity.copy()
                        new_entity.dxf.layer = layer_name
                        
                        # Use the ACTUAL center that was used during expansion (not simple width/2, height/2)
                        if hasattr(tool, 'actual_center_x_mm') and hasattr(tool, 'actual_center_y_mm'):
                            # For DXF tools with expansion, use the actual geometric center
                            tool_center_x = tool.actual_center_x_mm
                            tool_center_y = tool.actual_center_y_mm
                        else:
                            # For custom shapes (no expansion), use simple center
                            tool_center_x = tool.width_mm / 2
                            tool_center_y = tool.height_mm / 2
                        
                        # ===== TRANSFORMATION ORDER: FLIP → ROTATE → TRANSLATE =====
                        
                        # STEP 1: Apply FLIPS first (around tool center)
                        if tool.flip_horizontal or tool.flip_vertical:
                            # Translate to origin for flipping
                            if hasattr(new_entity, 'translate'):
                                new_entity.translate(-tool_center_x, -tool_center_y, 0)
                            else:
                                self._translate_entity_manually(new_entity, -tool_center_x, -tool_center_y, 0)
                            
                            # Apply horizontal flip
                            if tool.flip_horizontal:
                                self._flip_entity_horizontal(new_entity, 0)  # Flip around x=0 (now at origin)
                            
                            # Apply vertical flip
                            if tool.flip_vertical:
                                self._flip_entity_vertical(new_entity, 0)  # Flip around y=0 (now at origin)
                            
                            # Translate back to tool space
                            if hasattr(new_entity, 'translate'):
                                new_entity.translate(tool_center_x, tool_center_y, 0)
                            else:
                                self._translate_entity_manually(new_entity, tool_center_x, tool_center_y, 0)
                        
                        # STEP 2: Apply ROTATION (around ACTUAL center)
                        if tool.rotation_degrees != 0:
                            import math
                            
                            # Negate rotation to fix DXF coordinate system vs canvas coordinate system
                            corrected_rotation = -tool.rotation_degrees
                            
                            # Translate to origin for rotation (using ACTUAL center)
                            if hasattr(new_entity, 'translate'):
                                new_entity.translate(-tool_center_x, -tool_center_y, 0)
                            else:
                                self._translate_entity_manually(new_entity, -tool_center_x, -tool_center_y, 0)
                            
                            # Rotate around origin
                            if hasattr(new_entity, 'rotate_z'):
                                new_entity.rotate_z(math.radians(corrected_rotation))
                            else:
                                self._rotate_entity_manually(new_entity, corrected_rotation, 0, 0)
                            
                            # Translate back to tool space (using ACTUAL center)
                            if hasattr(new_entity, 'translate'):
                                new_entity.translate(tool_center_x, tool_center_y, 0)
                            else:
                                self._translate_entity_manually(new_entity, tool_center_x, tool_center_y, 0)
                        
                        # STEP 3: TRANSLATE to final canvas position
                        if hasattr(new_entity, 'translate'):
                            new_entity.translate(offset_x_mm, offset_y_mm, offset_z_mm)
                        else:
                            self._translate_entity_manually(new_entity, offset_x_mm, offset_y_mm, offset_z_mm)
                        
                        # SET DXF ATTRIBUTES FOR CNC MACHINE
                        new_entity.dxf.layer = layer_name
                        
                        # ADD ELEVATION (Z-height where cutting starts)
                        if hasattr(new_entity.dxf, 'elevation'):
                            new_entity.dxf.elevation = offset_z_mm
                        
                        # ADD THICKNESS (cutting depth - extrusion direction)
                        if hasattr(new_entity.dxf, 'thickness'):
                            new_entity.dxf.thickness = cut_depth_mm
                        
                        new_entity.dxf.color = 1  # Red for all tool cutouts
                        
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
            
            s3_url = None
            if upload_to_s3:
                s3_url = self.s3_manager.upload_file(local_output_path, output_filename, 'application/dxf')
            
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

@app.route('/', methods=['GET'])
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

@app.route('/api/health', methods=['GET'])
def api_health():
    try:
        s3_manager = S3Manager()
        return jsonify({
            "status": "healthy",
            "s3_available": s3_manager.s3_client is not None
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/compose', methods=['POST'])
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

def smooth_contour_preserve_corners(contour, angle_threshold=10.0, smooth_window=3):
    """
    Smooth contour while preserving sharp corners.
    Now includes MICRO-LEVEL corner smoothing for better CNC performance.
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
    
    # Apply moving average only to non-corner points
    for i in range(n):
        if not is_corner[i]:
            # Get neighboring points (excluding corners)
            neighbors = []
            for offset in range(-smooth_window, smooth_window + 1):
                idx = (i + offset) % n
                if not is_corner[idx]:
                    neighbors.append(contour[idx])
            
            if len(neighbors) > 1:
                smoothed[i] = np.mean(neighbors, axis=0)
    
    # ========== NEW: MICRO-LEVEL CORNER SMOOTHING ==========
    # Apply TINY smoothing at corner points for better CNC performance
    # This creates a micro-radius that's invisible at macro level
    corner_smooth_radius = 1  # Number of adjacent points to include (1 = 3 points total)
    corner_smooth_weight = 0.15  # How much to smooth (0.15 = 15% blend, keeps 85% sharp)
    
    for i in range(n):
        if is_corner[i]:
            # Get immediate neighbors (just 1 point on each side)
            p_prev = smoothed[(i - corner_smooth_radius) % n]
            p_curr = smoothed[i]
            p_next = smoothed[(i + corner_smooth_radius) % n]
            
            # Apply weighted average - mostly keep original, slight blend with neighbors
            # This creates a tiny fillet radius at the corner
            smoothed[i] = (
                (1 - 2 * corner_smooth_weight) * p_curr +  # 70% original
                corner_smooth_weight * p_prev +             # 15% previous
                corner_smooth_weight * p_next               # 15% next
            )
    # ========== END MICRO-LEVEL CORNER SMOOTHING ==========
    
    return smoothed.reshape(-1, 1, 2).astype(np.int32)


def adaptive_smooth_contour(contour, preserve_sharp_angles=True):
    """
    Multi-stage adaptive smoothing that preserves geometry while removing zig-zags.
    
    This is the main function to replace Gaussian smoothing.
    """
    # Stage 1: Remove points that are too close (< 2 pixels apart)
    points = contour.reshape(-1, 2)
    points = remove_close_points(points, min_distance=1.5)
    
    # Stage 2: Light Douglas-Peucker to remove micro-variations
    # Very small epsilon to only remove pixel-level noise
    contour_temp = points.reshape(-1, 1, 2).astype(np.int32)
    epsilon_micro = 0.5  # pixels - only removes sub-pixel variations
    contour_temp = cv2.approxPolyDP(contour_temp, epsilon_micro, True)
    
    if not preserve_sharp_angles:
        return contour_temp
    
    # Stage 3: Angle-based smoothing - smooth straight segments, preserve corners
    smoothed = smooth_contour_preserve_corners(
        contour_temp, 
        angle_threshold=10.0,  # Angles > 10° from straight are preserved
        smooth_window=2         # Small window for subtle smoothing
    )
    
    return smoothed
#---------------------------------------------------------
@app.route('/api/image-to-dxf', methods=['POST'])
def api_image_to_dxf():
    """
    Convert cutout tool image to DXF file with contours
    Expected form data:
    - image: file (PNG/JPG with transparent or white background)
    - length: float (tool length)
    - depth: float (tool depth)
    - unit: string ('mm' or 'inches')
    """
    logger.info("=== Incoming Request Data ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Content-Type: {request.content_type}")
    
    # Log form data
    logger.info("Form Data:")
    for key, value in request.form.items():
        logger.info(f"  {key}: {value}")
    
    # Log files
    logger.info("Files:")
    for key, file in request.files.items():
        logger.info(f"  {key}: {file.filename} (content_type: {file.content_type})")
    
    # Log headers (optional - can be verbose)
    # logger.info("Headers:")
    # for key, value in request.headers.items():
    #     logger.info(f"  {key}: {value}")
    
    logger.info("============================")

    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        # Get parameters
        try:
            length_value = float(request.form.get('length', 0))
            depth_value = float(request.form.get('depth', 0))
            unit = request.form.get('unit', 'inches').lower()
 
            # Convert mm to inches if needed
            if unit == 'mm':
                length_inches = length_value / 25.4
                depth_inches = depth_value / 25.4
                print(f"Lenghth in mm : {length_inches}________________________________________________________")
                logger.info(f"Converted from mm: {length_value}mm -> {length_inches:.3f} inches")
            elif unit == 'inches':
                length_inches = length_value
                depth_inches = depth_value
                print(f"Lenghth in inches : {length_inches}________________________________________________________")
            else:
                return jsonify({"success": False, "error": "Invalid unit. Must be 'mm' or 'inches'"}), 400
                
        except ValueError:
            return jsonify({"success": False, "error": "Invalid numeric parameters"}), 400
        
        if length_inches <= 0:
            return jsonify({"success": False, "error": "Length must be positive"}), 400
        
        logger.info(f"Processing image: {image_file.filename}, {length_inches:.3f} inches (depth: {depth_inches:.3f} inches)")
        
        # Read image
        image_bytes = image_file.read()
        image_bytes = compress_image_if_needed(image_bytes, max_size_mb=2)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        
        if img is None:
            return jsonify({"success": False, "error": "Invalid image format"}), 400
        
        # Convert to grayscale and create mask
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                # Use alpha channel as mask
                mask = img[:, :, 3]
            else:  # RGB
                # Convert to grayscale and threshold
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        else:
            # Already grayscale
            _, mask = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
            
        # mask = cv2.medianBlur(mask, 3)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return jsonify({"success": False, "error": "No contours found in image"}), 400
        
        # Get the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
# ====== ADVANCED SMOOTHING: Preserve sharp features, smooth micro zig-zags ======
        # This replaces the old Gaussian smoothing that was too aggressive
        
        # Apply adaptive smoothing that preserves sharp corners
        main_contour = adaptive_smooth_contour(
            main_contour, 
            preserve_sharp_angles=True
        )
        
        # Optional: Very light final approximation (if needed)
        # This removes any remaining sub-pixel noise without affecting geometry
        epsilon_final = 0.0001 * cv2.arcLength(main_contour, True)
        main_contour = cv2.approxPolyDP(main_contour, epsilon_final, True)
        # ====== END ADVANCED SMOOTHING ======
        
        # === APPLY CONFIGURABLE TOOL OFFSET FIRST ==================================================
        tool_offset_inches = get_tool_offset_inches()
        logger.info(f"Applying tool offset: {tool_offset_inches} inches")
        
        # Get original image dimensions
        img_height, img_width = img.shape[:2]
        
        # Calculate TEMPORARY scale factor to convert offset from inches to pixels
        # (based on user's entered dimension)
        temp_scale_factor = img_height / length_inches  # pixels per inch
        
        # Convert offset from inches to pixels
        # Convert offset from inches to pixels using config value
        offset_pixels = tool_offset_inches * temp_scale_factor
        # Convert contour to Shapely polygon for buffer operation
        points_for_offset = [(p[0][0], p[0][1]) for p in main_contour]
        poly_original = Polygon(points_for_offset)
        
        # Validate and fix if needed
        if not poly_original.is_valid:
            poly_original = poly_original.buffer(0)
        
        # Apply outward buffer (expansion)
        poly_with_offset = poly_original.buffer(offset_pixels)
        
        # Handle MultiPolygon case
        if poly_with_offset.geom_type == 'MultiPolygon':
            poly_with_offset = max(poly_with_offset.geoms, key=lambda p: p.area)
        
        # Convert back to numpy array for OpenCV
        offset_contour_coords = np.array(poly_with_offset.exterior.coords, dtype=np.float32)
        main_contour = offset_contour_coords.reshape((-1, 1, 2)).astype(np.int32)
        # === END TOOL OFFSET =============================================================================
        
        # scale_factor is calculated later, after h_main is known (line ~2431)

        
        # Calculate pixels for 0.05 inches contour thickness
        contour_thickness_inches = 0.05
        contour_thickness_pixels = int((contour_thickness_inches / length_inches) * img_height)
        
        # Calculate pixels for 0.5 inches expansion (previously)
        expansion_inches = 0.25
        expansion_pixels = (expansion_inches / length_inches) * img_height
        
        # Convert main contour to Shapely Polygon
        points = [(p[0][0], p[0][1]) for p in main_contour]
        poly = Polygon(points)
        
        # Validate polygon
        if not poly.is_valid:
            # Try to fix invalid polygon
            poly = poly.buffer(0)
        
        # Apply buffer for expansion (positive = expand)
        expanded_poly = poly.buffer(expansion_pixels)
        
        # Handle MultiPolygon case (if buffer operation creates multiple polygons)
        if expanded_poly.geom_type == 'MultiPolygon':
            # Get the largest polygon from the multipolygon
            expanded_poly = max(expanded_poly.geoms, key=lambda p: p.area)
        
        # Convert back to numpy arrays for OpenCV
        main_contour_coords = np.array(poly.exterior.coords, dtype=np.float32)
        main_contour_smooth = main_contour_coords.reshape((-1, 1, 2)).astype(np.int32)
        
        expanded_contour_coords = np.array(expanded_poly.exterior.coords, dtype=np.float32)
        expanded_contour_smooth = expanded_contour_coords.reshape((-1, 1, 2)).astype(np.int32)
        
        # Calculate the bounding boxes of both contours to determine required canvas size
        x_main, y_main, w_main, h_main = cv2.boundingRect(main_contour_smooth)
        x_exp, y_exp, w_exp, h_exp = cv2.boundingRect(expanded_contour_smooth)
        
        # Calculate the overall bounding box that contains both contours
        x_min = min(x_main, x_exp)
        y_min = min(y_main, y_exp)
        x_max = max(x_main + w_main, x_exp + w_exp)
        y_max = max(y_main + h_main, y_exp + h_exp)
        
        # Calculate required expansion to fit both contours with padding
        padding = 1  # Minimum padding in pixels
        
        # Calculate how much we need to expand each side
        left_expansion = max(0, padding - x_min)
        top_expansion = max(0, padding - y_min)
        right_expansion = max(0, (x_max + padding) - img_width)
        bottom_expansion = max(0, (y_max + padding) - img_height)
        
        # Calculate new dimensions
        new_width = img_width + left_expansion + right_expansion
        new_height = img_height + top_expansion + bottom_expansion
        
        # Calculate offset to center the original image in the new canvas
        offset_x = left_expansion
        offset_y = top_expansion
        
        # Create DXF file
        doc = ezdxf.new(units=ezdxf.units.MM)
        doc.header["$INSUNITS"] = ezdxf.units.MM
        msp = doc.modelspace()



        # Get bounding box of main contour to normalize it------------------------------------------------------------------------------
        x_main, y_main, w_main, h_main = cv2.boundingRect(main_contour_smooth)

        # CORRECT scale_factor: maps h_main pixels → exactly (length + 2*offset) inches
        # h_main already includes the offset buffer, so this formula is mathematically exact
        # DXF = h_main * scale_factor = (length + 2*offset) * 25.4 mm — always correct
        scale_factor = ((length_inches + 2 * tool_offset_inches) * 25.4) / h_main
        
        # Convert main contour to DXF coordinates (normalized to start at 0,0)
        dxf_points = []
        for point in main_contour_smooth:
            x_px, y_px = point[0]
            # Normalize to bounding box origin (translate to 0,0)
            x_normalized = x_px - x_main
            y_normalized = y_px - y_main
            # Convert to mm and flip Y axis (image coords vs DXF coords)
            x_mm = x_normalized * scale_factor
            y_mm = (h_main - y_normalized) * scale_factor
            dxf_points.append((x_mm, y_mm))
        
        # ========== CONFIGURABLE SPLINE SMOOTHNESS ==========
        # Control how smooth the curves are at different zoom levels
        # Options:
        #   "polyline" = straight segments (original, no smoothing)
        #   "minimal" = barely visible smoothing (micro-level only, degree=2)
        #   "light" = light curves (visible at 200% zoom, degree=2 with more points)
        #   "medium" = moderate curves (visible at 100% zoom, degree=3)
        #   "smooth" = strong curves (visible at macro level, degree=3 dense)
        
        SPLINE_SMOOTHNESS = "minimal"  # ← CHANGE THIS VALUE TO ADJUST
        
        if SPLINE_SMOOTHNESS == "polyline":
            # Original: straight line segments (no spline)
            msp.add_lwpolyline(dxf_points, close=True, dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})
        
        elif SPLINE_SMOOTHNESS == "minimal":
            # Quadratic spline (degree=2) - barely visible curves, only at micro level
            # Uses all points, minimal deviation from straight lines
            spline_points = dxf_points + [dxf_points[0]]
            msp.add_spline(fit_points=spline_points, degree=2, dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})
        
        elif SPLINE_SMOOTHNESS == "light":
            # Quadratic spline with point reduction - light curves visible at ~200% zoom
            # Skip every other point to allow more curve freedom
            reduced_points = dxf_points[::2]  # Take every 2nd point
            if len(reduced_points) < 4:
                reduced_points = dxf_points  # Too few points, use all
            spline_points = reduced_points + [reduced_points[0]]
            msp.add_spline(fit_points=spline_points, degree=2, dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})
        
        elif SPLINE_SMOOTHNESS == "medium":
            # Cubic spline with moderate point reduction - visible at ~100% zoom
            # Skip every 3rd point
            reduced_points = dxf_points[::3]
            if len(reduced_points) < 4:
                reduced_points = dxf_points
            spline_points = reduced_points + [reduced_points[0]]
            msp.add_spline(fit_points=spline_points, degree=3, dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})
        
        elif SPLINE_SMOOTHNESS == "smooth":
            # Cubic spline with aggressive point reduction - strong visible curves at macro level
            # Skip every 5th point for maximum smoothing
            reduced_points = dxf_points[::5]
            if len(reduced_points) < 4:
                reduced_points = dxf_points[::2]
            spline_points = reduced_points + [reduced_points[0]]
            msp.add_spline(fit_points=spline_points, degree=3, dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})
        
        else:
            # Default fallback to minimal
            spline_points = dxf_points + [dxf_points[0]]
            msp.add_spline(fit_points=spline_points, degree=2, dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})
        
        # Convert main contour to DXF coordinates
        # dxf_points = []
        # for point in main_contour_smooth:
        #     x_px, y_px = point[0]
        #     # Convert to mm and flip Y axis (image coords vs DXF coords)
        #     x_mm = x_px * scale_factor
        #     y_mm = (img_height - y_px) * scale_factor
        #     dxf_points.append((x_mm, y_mm))
        
        # # Add main contour as polyline to DXF
        # msp.add_lwpolyline(dxf_points, close=True, dxfattribs={"layer": "TOOL_CONTOUR", "color": 1})
        
        
        # Save DXF
        dxf_filename = f"tool_{uuid.uuid4().hex[:8]}.dxf"
        dxf_path = os.path.join(HF_CACHE_DIR, dxf_filename)
        doc.saveas(dxf_path)
        
        
        # Create expanded canvas with transparency for merged image
        img_merged_canvas = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        
        # Place original image in the expanded canvas
        if len(img.shape) == 2:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = 255  # Full opacity
        else:
            img_rgba = img.copy()
        
        # Place the original image in the correct position within the expanded canvas
        img_merged_canvas[offset_y:offset_y+img_height, offset_x:offset_x+img_width] = img_rgba
        
        # Offset contours to match new canvas coordinates
        main_contour_offset = main_contour_smooth + [offset_x, offset_y]
        expanded_contour_offset = expanded_contour_smooth + [offset_x, offset_y]
        
        # === MERGED IMAGE: Show both contours on the same image ===
        
        # 1. Create a SOLID gray background for the expanded area
        gray_background = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        # Fill the ENTIRE expanded contour area with gray
        cv2.fillPoly(gray_background, [expanded_contour_offset], (194, 194, 194, 255))
        
        # 2. Composite: gray background first, then tool on top
        # Where tool has transparency (alpha < 255), gray shows through
        alpha_tool = img_merged_canvas[:, :, 3] / 255.0
        alpha_gray = gray_background[:, :, 3] / 255.0 * (1.0 - alpha_tool)
        
        for c in range(3):
            img_merged_canvas[:, :, c] = (alpha_tool * img_merged_canvas[:, :, c] + 
                                           alpha_gray * gray_background[:, :, c])
        img_merged_canvas[:, :, 3] = np.maximum(img_merged_canvas[:, :, 3], gray_background[:, :, 3])
        
        # 3. Draw the blue/gold contour outline on the original tool (strict contour)
        cv2.polylines(img_merged_canvas, [main_contour_offset], True, (168, 108, 38, 255), contour_thickness_pixels, cv2.LINE_AA)
        
        # 4. Draw the outer expanded contour outline in gray
        cv2.polylines(img_merged_canvas, [expanded_contour_offset], True, (100, 100, 100, 255), 1, cv2.LINE_AA)

                
        # Save merged annotated image
        merged_filename = f"tool_merged_{uuid.uuid4().hex[:8]}.png"
        merged_path = os.path.join(HF_CACHE_DIR, merged_filename)
        cv2.imwrite(merged_path, img_merged_canvas)

        # Save original image
        original_img_filename = f"original_img_{uuid.uuid4().hex[:8]}.png"
        original_img_path = os.path.join(HF_CACHE_DIR, original_img_filename)
        
        # Convert original image to RGBA if needed for consistency
        if len(img.shape) == 2:
            img_original_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img_original_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_original_rgba[:, :, 3] = 255  # Full opacity
        else:
            img_original_rgba = img.copy()
        
        cv2.imwrite(original_img_path, img_original_rgba)

        # Create simple contour image (just tight blue contour on transparent background)
        simple_contour_filename = f"simple_contour_{uuid.uuid4().hex[:8]}.png"
        simple_contour_path = os.path.join(HF_CACHE_DIR, simple_contour_filename)
        
        # Create a transparent canvas
        img_simple_contour = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        
        # Place the original image in the expanded canvas
        if len(img.shape) == 2:
            img_rgba_simple = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img_rgba_simple = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_rgba_simple[:, :, 3] = 255  # Full opacity
        else:
            img_rgba_simple = img.copy()
        
        # Place the original image in the correct position
        img_simple_contour[offset_y:offset_y+img_height, offset_x:offset_x+img_width] = img_rgba_simple
        
        # Draw only the tight blue/gold contour outline
        cv2.polylines(img_simple_contour, [main_contour_offset], True, (168, 108, 38, 255), contour_thickness_pixels, cv2.LINE_AA)
        
        cv2.imwrite(simple_contour_path, img_simple_contour)
        
        
        # Upload to S3
        s3_manager = S3Manager()
        
        dxf_url = s3_manager.upload_file(dxf_path, dxf_filename, 'application/dxf')
        simple_img_url = s3_manager.upload_file(original_img_path, original_img_filename, 'image/png')
        merged_url = s3_manager.upload_file(merged_path, merged_filename, 'image/png')
        
        # Cleanup local files
        for path in [dxf_path, simple_contour_path, merged_path]:
            if os.path.exists(path):
                os.remove(path)
        
        if not all([dxf_url, simple_img_url, merged_url]):
            return jsonify({
                "success": False,
                "error": "Failed to upload files to S3"
            }), 500
        
        return jsonify({
            "success": True,
            #"dxf_url": dxf_url,
            "original_img" : simple_img_url,
            "contour_image_url": merged_url
        }), 200
        
    except Exception as e:
        logger.error(f"Image to DXF conversion failed: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def compress_image_if_needed(image_bytes, max_size_mb=2):
    """
    Compress image to fit within max_size_mb
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if len(image_bytes) <= max_size_bytes:
        return image_bytes
    
    logger.info(f"Image size {len(image_bytes)/1024/1024:.2f}MB exceeds {max_size_mb}MB, compressing...")
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return image_bytes
    
    # Start with quality 95 and reduce until size is acceptable
    quality = 95
    while quality > 10:
        _, compressed = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, quality // 10])
        if len(compressed) <= max_size_bytes:
            logger.info(f"Compressed to {len(compressed)/1024/1024:.2f}MB with quality {quality}")
            return compressed.tobytes()
        quality -= 10
    
    # If still too large, resize the image
    scale_factor = (max_size_bytes / len(image_bytes)) ** 0.5
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    _, compressed = cv2.imencode('.png', img_resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    logger.info(f"Resized and compressed to {len(compressed)/1024/1024:.2f}MB")
    return compressed.tobytes()
    
        
@app.route('/api/status', methods=['GET'])
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

@app.route('/api/tool-offset', methods=['GET', 'POST'])
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
        "available_endpoints": ["/", "/api/health", "/api/compose", "/api/status"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    logger.info(f"Starting DXF Canvas Composer on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
