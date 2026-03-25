import ezdxf
import requests
import tempfile
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from urllib.parse import urlparse
import uuid
import json
from dataclasses import dataclass, asdict
import boto3
from botocore.exceptions import ClientError
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS S3 Configuration
S3_BUCKET_NAME = "axionbucketmlb"
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')

# Hugging Face cache directory
HF_CACHE_DIR = os.getenv('HF_HOME', '/tmp/hf_cache')
os.makedirs(HF_CACHE_DIR, exist_ok=True)

@dataclass
class ToolInfo:
    """Data class to store tool information"""
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

@dataclass
class CanvasConfig:
    """Canvas configuration"""
    width_inches: float
    height_inches: float
    thickness_inches: float
    has_overlaps: bool = False

@dataclass
class LayoutMetadata:
    """Layout metadata"""
    layout_name: str
    brand: str
    container_type: str

class S3Manager:
    """AWS S3 operations manager"""
    
    def __init__(self):
        """Initialize S3 client with credentials from environment"""
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', AWS_REGION)
        
        if not self.aws_access_key or not self.aws_secret_key:
            logger.warning("AWS credentials not found in environment variables")
            self.s3_client = None
        else:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                logger.info("S3 client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self.s3_client = None
    
    def upload_file(self, file_path: str, filename: str, content_type: str = None) -> Optional[str]:
        """
        Upload file to AWS S3 with proper content type for web preview
        
        Args:
            file_path: Local file path to upload
            filename: Desired filename in S3
            content_type: MIME type for the file
            
        Returns:
            Public URL of uploaded file or None if failed
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
        
        try:
            # Generate unique filename to prevent collisions
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            
            # Set proper content type and disposition for web preview
            extra_args = {
                'ContentDisposition': 'inline'  # View in browser, not download
            }
            
            if content_type:
                extra_args['ContentType'] = content_type
            
            # Upload file with metadata
            self.s3_client.upload_file(
                file_path,
                S3_BUCKET_NAME,
                unique_filename,
                ExtraArgs=extra_args
            )
            
            # Generate public URL
            public_url = f"https://{S3_BUCKET_NAME}.s3.{self.aws_region}.amazonaws.com/{unique_filename}"
            
            logger.info(f"File uploaded to S3: {public_url}")
            return public_url
            
        except ClientError as e:
            logger.error(f"AWS S3 upload failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return None
    
    def download_file(self, url: str, local_path: str) -> bool:
        """
        Download file from S3 or HTTP URL
        
        Args:
            url: Source URL (S3 or HTTP)
            local_path: Destination local path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Check if it's an S3 URL
            if 's3.amazonaws.com' in url or S3_BUCKET_NAME in url:
                # Extract S3 key from URL
                parsed = urlparse(url)
                s3_key = parsed.path.lstrip('/')
                
                if self.s3_client:
                    self.s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
                    logger.info(f"Downloaded from S3: {url}")
                    return True
            
            # Fall back to HTTP download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded from HTTP: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False


class DXFCanvasComposer:
    """
    Industrial-scale DXF canvas composition system with AWS S3 integration
    Designed for Hugging Face deployment
    """
    
    def __init__(self):
        """Initialize the composer with S3 manager and cache directory"""
        self.s3_manager = S3Manager()
        self.cache_dir = os.path.join(HF_CACHE_DIR, f"dxf_cache_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"DXF cache directory: {self.cache_dir}")
        
    def __del__(self):
        """Cleanup temporary files"""
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove cache directory and all contents"""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleaned up cache directory: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup cache: {e}")
    
    def _download_dxf_file(self, url: str, tool_id: str) -> Optional[str]:
        """
        Download DXF file from URL to cache
        
        Args:
            url: DXF file URL (AWS S3 or HTTP)
            tool_id: Unique tool identifier
            
        Returns:
            Local file path or None if failed
        """
        try:
            # Generate safe filename
            filename = f"{tool_id}_{uuid.uuid4().hex[:6]}.dxf"
            local_path = os.path.join(self.cache_dir, filename)
            
            # Download file
            if self.s3_manager.download_file(url, local_path):
                # Verify it's a valid DXF file
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    try:
                        # Quick validation
                        doc = ezdxf.readfile(local_path)
                        logger.info(f"Valid DXF downloaded: {local_path}")
                        return local_path
                    except Exception as e:
                        logger.error(f"Invalid DXF file from {url}: {e}")
                        os.remove(local_path)
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to download DXF from {url}: {e}")
            return None
    
    def _extract_entity_points(self, entity):
        """Extract all relevant points from an entity for bounds calculation"""
        points = []
        
        try:
            if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'start'):
                # Line entities
                points = [entity.dxf.start, entity.dxf.end]
            elif hasattr(entity, 'vertices'):
                # Polyline entities - handle both POLYLINE and LWPOLYLINE
                try:
                    # Try POLYLINE first
                    points = [vertex.dxf.location for vertex in entity.vertices()]
                except:
                    # LWPOLYLINE handling
                    try:
                        vertices_data = list(entity.vertices())
                        points = [(v[0], v[1], 0) if len(v) >= 2 else (0, 0, 0) for v in vertices_data]
                    except:
                        points = []
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'center'):
                # Circle/Arc entities
                center = entity.dxf.center
                radius = getattr(entity.dxf, 'radius', 0)
                points = [
                    (center.x - radius, center.y - radius, center.z),
                    (center.x + radius, center.y + radius, center.z)
                ]
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'insert'):
                # Insert/Block entities
                points = [entity.dxf.insert]
        except Exception as e:
            logger.warning(f"Failed to extract points from {type(entity)}: {e}")
            points = []
        
        return points

    def _analyze_and_normalize_dxf(self, dxf_path: str) -> Tuple[float, float, float, List]:
        """
        Analyze DXF file and return normalized entities
        
        Returns:
            Tuple of (width_mm, height_mm, depth_mm, normalized_entities)
        """
        try:
            logger.info(f"Analyzing and normalizing DXF: {dxf_path}")
            
            # Open DXF file
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            # Get all entities
            entities = list(msp)
            
            if not entities:
                logger.warning(f"No entities found in DXF: {dxf_path}")
                return 0.0, 0.0, 0.0, []
            
            # Calculate bounding box FIRST
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            for entity in entities:
                try:
                    points = self._extract_entity_points(entity)
                    
                    # Update bounds
                    for point in points:
                        if hasattr(point, 'x'):
                            x, y, z = point.x, point.y, getattr(point, 'z', 0)
                        elif isinstance(point, (tuple, list)) and len(point) >= 2:
                            x, y = point[0], point[1]
                            z = point[2] if len(point) > 2 else 0
                        else:
                            continue
                            
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
                        min_z = min(min_z, z)
                        max_z = max(max_z, z)
                
                except Exception as e:
                    logger.warning(f"Error processing entity {type(entity)}: {e}")
                    continue
            
            # Calculate dimensions
            width_mm = max_x - min_x if min_x != float('inf') else 0.0
            height_mm = max_y - min_y if min_y != float('inf') else 0.0
            depth_mm = max(max_z - min_z, 1.0) if min_z != float('inf') else 1.0
            
            # NOW normalize all entities to start from (0,0)
            normalized_entities = []
            for entity in entities:
                try:
                    normalized_entity = entity.copy()
                    # Move entity so its bounding box starts at origin
                    self._translate_entity_manually(normalized_entity, -min_x, -min_y, -min_z)
                    normalized_entities.append(normalized_entity)
                except Exception as e:
                    logger.warning(f"Failed to normalize entity {type(entity)}: {e}")
                    normalized_entities.append(entity.copy())
            
            logger.info(f"DXF normalized: {width_mm:.2f}mm x {height_mm:.2f}mm x {depth_mm:.2f}mm")
            
            return width_mm, height_mm, depth_mm, normalized_entities
            
        except Exception as e:
            logger.error(f"Failed to analyze DXF {dxf_path}: {e}")
            return 0.0, 0.0, 0.0, []
    
    def _translate_entity_manually(self, entity, dx: float, dy: float, dz: float = 0):
        """Manually translate entity coordinates"""
        try:
            if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                # Line entity
                entity.dxf.start = (entity.dxf.start.x + dx, entity.dxf.start.y + dy, entity.dxf.start.z + dz)
                entity.dxf.end = (entity.dxf.end.x + dx, entity.dxf.end.y + dy, entity.dxf.end.z + dz)
                
            elif hasattr(entity.dxf, 'center'):
                # Circle/Arc entity
                entity.dxf.center = (entity.dxf.center.x + dx, entity.dxf.center.y + dy, entity.dxf.center.z + dz)
                
            elif hasattr(entity.dxf, 'insert'):
                # Block/Insert entity
                entity.dxf.insert = (entity.dxf.insert.x + dx, entity.dxf.insert.y + dy, entity.dxf.insert.z + dz)
                
            elif hasattr(entity, 'vertices'):
                # Handle both POLYLINE and LWPOLYLINE
                entity_type = type(entity).__name__
                
                if entity_type == 'LWPolyline':
                    try:
                        current_vertices = list(entity.vertices())
                        new_vertices = []
                        
                        for vertex in current_vertices:
                            if len(vertex) >= 2:
                                new_x = vertex[0] + dx
                                new_y = vertex[1] + dy
                                if len(vertex) > 2:
                                    new_vertices.append((new_x, new_y, *vertex[2:]))
                                else:
                                    new_vertices.append((new_x, new_y))
                            else:
                                new_vertices.append(vertex)
                        
                        entity.clear()
                        for vertex in new_vertices:
                            if len(vertex) >= 2:
                                entity.append(vertex[:2])
                        
                    except Exception as e:
                        logger.warning(f"LWPOLYLINE translation failed: {e}")
                        try:
                            entity.translate(dx, dy, dz)
                        except:
                            pass
                            
                else:
                    # Regular POLYLINE
                    try:
                        for vertex in entity.vertices():
                            if hasattr(vertex.dxf, 'location'):
                                loc = vertex.dxf.location
                                vertex.dxf.location = (loc.x + dx, loc.y + dy, loc.z + dz)
                    except Exception as e:
                        logger.warning(f"POLYLINE translation failed: {e}")
            
        except Exception as e:
            logger.warning(f"Manual translation failed for {type(entity)}: {e}")
            try:
                if hasattr(entity, 'translate'):
                    entity.translate(dx, dy, dz)
            except:
                pass
    
    def _mm_to_inches(self, mm: float) -> float:
        """Convert millimeters to inches"""
        return mm / 25.4
    
    def _inches_to_mm(self, inches: float) -> float:
        """Convert inches to millimeters"""
        return inches * 25.4
    
    def _validate_canvas_fit(self, tools: List[ToolInfo], canvas: CanvasConfig) -> bool:
        """Validate that all tools fit within canvas boundaries"""
        canvas_width_mm = self._inches_to_mm(canvas.width_inches)
        canvas_height_mm = self._inches_to_mm(canvas.height_inches)
        canvas_thickness_mm = self._inches_to_mm(canvas.thickness_inches)
        
        for tool in tools:
            pos_x_mm = self._inches_to_mm(tool.position_x_inches)
            pos_y_mm = self._inches_to_mm(tool.position_y_inches)
            
            if (pos_x_mm + tool.width_mm > canvas_width_mm or
                pos_y_mm + tool.height_mm > canvas_height_mm or
                tool.depth_mm > canvas_thickness_mm):
                
                logger.error(f"Tool '{tool.name}' extends beyond canvas boundaries")
                return False
        
        return True
    
    def _check_tool_overlaps(self, tools: List[ToolInfo], min_spacing_mm: float = 2.54) -> List[str]:
        """Check for overlapping tools"""
        warnings = []
        
        for i, tool1 in enumerate(tools):
            for j, tool2 in enumerate(tools[i+1:], i+1):
                box1 = box(
                    self._inches_to_mm(tool1.position_x_inches),
                    self._inches_to_mm(tool1.position_y_inches),
                    self._inches_to_mm(tool1.position_x_inches) + tool1.width_mm,
                    self._inches_to_mm(tool1.position_y_inches) + tool1.height_mm
                )
                
                box2 = box(
                    self._inches_to_mm(tool2.position_x_inches),
                    self._inches_to_mm(tool2.position_y_inches),
                    self._inches_to_mm(tool2.position_x_inches) + tool2.width_mm,
                    self._inches_to_mm(tool2.position_y_inches) + tool2.height_mm
                )
                
                if box1.intersects(box2):
                    warnings.append(f"Tools '{tool1.name}' and '{tool2.name}' overlap")
        
        return warnings
    
    def _add_3d_canvas_representation(self, msp, canvas: CanvasConfig):
        """Add 3D representation of canvas with thickness"""
        try:
            canvas_width_mm = self._inches_to_mm(canvas.width_inches)
            canvas_height_mm = self._inches_to_mm(canvas.height_inches)
            canvas_thickness_mm = self._inches_to_mm(canvas.thickness_inches)
            
            # Bottom face (z=0)
            msp.add_lwpolyline([
                (0, 0),
                (canvas_width_mm, 0),
                (canvas_width_mm, canvas_height_mm),
                (0, canvas_height_mm)
            ], close=True, dxfattribs={"layer": "CANVAS_BOTTOM", "color": 8})
            
            # Top face (z=thickness)
            for entity in msp.query('LWPOLYLINE[layer=="CANVAS_BOTTOM"]'):
                top_entity = entity.copy()
                top_entity.dxf.layer = "CANVAS_TOP"
                top_entity.dxf.color = 6
                self._translate_entity_manually(top_entity, 0, 0, canvas_thickness_mm)
                msp.add_entity(top_entity)
            
            # Vertical edges connecting top and bottom
            corners = [
                (0, 0),
                (canvas_width_mm, 0),
                (canvas_width_mm, canvas_height_mm),
                (0, canvas_height_mm)
            ]
            
            for corner in corners:
                msp.add_line(
                    (corner[0], corner[1], 0),
                    (corner[0], corner[1], canvas_thickness_mm),
                    dxfattribs={"layer": "CANVAS_EDGES", "color": 8}
                )
            
            logger.info(f"Added 3D canvas representation with thickness: {canvas_thickness_mm:.2f}mm")
            
        except Exception as e:
            logger.warning(f"Failed to add 3D canvas representation: {e}")
    
    def _add_metadata_to_dxf(self, msp, canvas: CanvasConfig, metadata: LayoutMetadata):
        """Add metadata as text entities to the DXF"""
        try:
            text_height = 2.0
            text_x = 5.0
            text_y = self._inches_to_mm(canvas.height_inches) - 10.0
            
            metadata_lines = [
                f"Layout: {metadata.layout_name}",
                f"Brand: {metadata.brand}",
                f"Type: {metadata.container_type}",
                f"Canvas: {canvas.width_inches}\" x {canvas.height_inches}\" x {canvas.thickness_inches}\"",
                f"3D Thickness: {self._inches_to_mm(canvas.thickness_inches):.2f}mm",
                f"Overlaps: {'Yes' if canvas.has_overlaps else 'No'}"
            ]
            
            for i, line in enumerate(metadata_lines):
                msp.add_text(
                    line,
                    dxfattribs={
                        "layer": "METADATA",
                        "height": text_height,
                        "color": 7
                    }
                ).set_pos((text_x, text_y - i * (text_height + 1)))
                
        except Exception as e:
            logger.warning(f"Failed to add metadata to DXF: {e}")
    
    def parse_layout_json(self, layout_data: Dict) -> Tuple[CanvasConfig, LayoutMetadata, List[ToolInfo]]:
        """Parse layout JSON data into structured objects"""
        try:
            # Parse canvas information
            canvas_info = layout_data.get("canvas_information", {})
            canvas = CanvasConfig(
                width_inches=canvas_info.get("width_inches", 21.0),
                height_inches=canvas_info.get("height_inches", 11.0),
                thickness_inches=canvas_info.get("thickness_inches", 0.5),
                has_overlaps=canvas_info.get("has_overlaps", False)
            )
            
            # Parse layout metadata
            layout_info = layout_data.get("layout_metadata", {})
            metadata = LayoutMetadata(
                layout_name=layout_info.get("layout_name", "Unknown"),
                brand=layout_info.get("brand", "Unknown"),
                container_type=layout_info.get("container_type", "Drawer")
            )
            
            # Parse tools
            tools_data = layout_data.get("tools", [])
            tools = []
            
            for tool_data in tools_data:
                dxf_link = tool_data.get("dxf_link", "")
                
                tool = ToolInfo(
                    tool_id=tool_data.get("tool_id", ""),
                    name=tool_data.get("name", "Unknown Tool"),
                    brand=tool_data.get("brand", "Unknown Brand"),
                    dxf_link=dxf_link,
                    position_x_inches=tool_data.get("position_inches", {}).get("x", 0.0),
                    position_y_inches=tool_data.get("position_inches", {}).get("y", 0.0),
                    rotation_degrees=tool_data.get("rotation_degrees", 0.0),
                    height_diagonal_inches=tool_data.get("height_diagonal_inches", 0.0),
                    thickness_inches=tool_data.get("thickness_inches", 0.5),
                    flip_horizontal=tool_data.get("flip_horizontal", False),
                    flip_vertical=tool_data.get("flip_vertical", False),
                    opacity=tool_data.get("opacity", 100),
                    smooth=tool_data.get("smooth", 0)
                )
                tools.append(tool)
            
            logger.info(f"Parsed layout: {len(tools)} tools on {canvas.width_inches}x{canvas.height_inches} canvas")
            return canvas, metadata, tools
            
        except Exception as e:
            logger.error(f"Failed to parse layout JSON: {e}")
            raise ValueError(f"Invalid layout JSON format: {str(e)}")

    def compose_canvas_from_json(self, layout_json: Dict, output_filename: str = None, upload_to_s3: bool = True) -> Dict:
        """
        Compose DXF canvas from JSON layout data with AWS S3 upload
        
        Args:
            layout_json: JSON data with complete layout information
            output_filename: Optional custom output filename
            upload_to_s3: Whether to upload result to S3
            
        Returns:
            Dict: Results with success status, S3 URL, and analysis
        """
        start_time = time.time()
        
        try:
            logger.info("=" * 80)
            logger.info("Starting DXF canvas composition from JSON")
            logger.info("=" * 80)
            
            # Parse JSON data
            canvas, metadata, tools = self.parse_layout_json(layout_json)
            
            if not tools:
                raise ValueError("No tools found in layout data")
            
            # Generate output filename if not provided
            if not output_filename:
                safe_name = "".join(c for c in metadata.layout_name if c.isalnum() or c in ('-', '_'))
                output_filename = f"{safe_name}_{metadata.brand}_{uuid.uuid4().hex[:8]}.dxf"
            
            # Download and process each DXF file (parallel downloads)
            logger.info(f"Downloading {len(tools)} DXF files from AWS S3...")
            
            download_tasks = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                for tool in tools:
                    future = executor.submit(self._download_dxf_file, tool.dxf_link, tool.tool_id)
                    download_tasks.append((tool, future))
            
            # Collect downloaded files and process them
            processed_tools = []
            for tool, future in download_tasks:
                try:
                    local_path = future.result(timeout=60)
                    
                    if not local_path:
                        raise Exception(f"Failed to download DXF for tool: {tool.name}")
                    
                    logger.info(f"Processing tool: {tool.name} ({tool.brand})")
                    
                    # Analyze dimensions with normalization
                    width_mm, height_mm, depth_mm, normalized_entities = self._analyze_and_normalize_dxf(local_path)
                    
                    if not normalized_entities:
                        raise Exception(f"No entities found in DXF for tool: {tool.name}")
                    
                    # Update tool with analyzed data
                    tool.width_mm = width_mm
                    tool.height_mm = height_mm
                    tool.depth_mm = depth_mm
                    tool.entities = normalized_entities
                    tool.local_path = local_path
                    
                    processed_tools.append(tool)
                    logger.info(f"✓ Tool processed: {tool.name} ({width_mm:.1f}x{height_mm:.1f}x{depth_mm:.1f}mm)")
                    
                except Exception as e:
                    logger.error(f"Failed to process tool {tool.name}: {e}")
                    raise Exception(f"Tool '{tool.name}' processing failed: {str(e)}")
            
            # Validate canvas fit
            logger.info("Validating canvas fit...")
            if not self._validate_canvas_fit(processed_tools, canvas):
                raise ValueError("One or more tools extend beyond canvas boundaries")
            
            # Check for overlaps
            if not canvas.has_overlaps:
                overlap_warnings = self._check_tool_overlaps(processed_tools)
                if overlap_warnings:
                    logger.warning("Overlaps detected in layout marked as non-overlapping")
            else:
                overlap_warnings = []
            
            # Create new DXF document with 3D canvas
            logger.info("Creating composed 3D DXF document...")
            doc = ezdxf.new(units=ezdxf.units.MM)
            doc.header["$INSUNITS"] = ezdxf.units.MM
            msp = doc.modelspace()
            
            # Add 3D canvas representation with thickness
            self._add_3d_canvas_representation(msp, canvas)
            
            # Add metadata
            self._add_metadata_to_dxf(msp, canvas, metadata)
            
            # Add each tool to the canvas
            placed_tools = []
            for i, tool in enumerate(processed_tools):
                try:
                    logger.info(f"Placing tool {i+1}/{len(processed_tools)}: {tool.name}")
                    
                    # Calculate offset in mm
                    offset_x_mm = self._inches_to_mm(tool.position_x_inches)
                    offset_y_mm = self._inches_to_mm(tool.position_y_inches)
                    
                    # Create layer for this tool
                    layer_name = f"TOOL_{i+1}_{tool.name.replace(' ', '_')}"
                    
                    # Add entities (they're already normalized to start at origin)
                    entities_added = 0
                    for entity in tool.entities:
                        try:
                            # Clone entity
                            new_entity = entity.copy()
                            
                            # Apply transformations FIRST (rotation, flips) around origin
                            self._apply_transformations(
                                new_entity,
                                tool.rotation_degrees,
                                tool.flip_horizontal,
                                tool.flip_vertical
                            )
                            
                            # THEN apply translation to final position
                            if hasattr(new_entity, 'translate'):
                                new_entity.translate(offset_x_mm, offset_y_mm, 0)
                            else:
                                self._translate_entity_manually(new_entity, offset_x_mm, offset_y_mm)
                            
                            # Set layer and attributes
                            new_entity.dxf.layer = layer_name
                            if tool.opacity < 100:
                                new_entity.dxf.color = max(1, int(256 * tool.opacity / 100))
                            
                            # Add to modelspace
                            msp.add_entity(new_entity)
                            entities_added += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to add entity {type(entity)} for tool {tool.name}: {e}")
                            continue
                    
                    # Add debug marker at tool position
                    msp.add_circle(
                        center=(offset_x_mm, offset_y_mm),
                        radius=2.0,
                        dxfattribs={"layer": f"{layer_name}_MARKER", "color": 1}
                    )
                    
                    placed_tools.append({
                        "tool_id": tool.tool_id,
                        "name": tool.name,
                        "brand": tool.brand,
                        "position_inches": (tool.position_x_inches, tool.position_y_inches),
                        "position_mm": (offset_x_mm, offset_y_mm),
                        "rotation_degrees": tool.rotation_degrees,
                        "dimensions_mm": (tool.width_mm, tool.height_mm, tool.depth_mm),
                        "dimensions_inches": (
                            self._mm_to_inches(tool.width_mm),
                            self._mm_to_inches(tool.height_mm),
                            self._mm_to_inches(tool.depth_mm)
                        ),
                        "height_diagonal_inches": tool.height_diagonal_inches,
                        "transformations": {
                            "flip_horizontal": tool.flip_horizontal,
                            "flip_vertical": tool.flip_vertical,
                            "opacity": tool.opacity,
                            "smooth": tool.smooth
                        },
                        "entities_count": entities_added,
                        "layer": layer_name
                    })
                    
                    logger.info(f"✓ Tool placed: {tool.name} at ({offset_x_mm:.1f}, {offset_y_mm:.1f})mm with {entities_added} entities")
                    
                except Exception as e:
                    logger.error(f"Failed to place tool {tool.name}: {e}")
                    raise Exception(f"Tool placement failed: {str(e)}")
            
            # Save composed DXF locally first
            output_dir = os.path.join(self.cache_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            local_output_path = os.path.join(output_dir, output_filename)
            
            doc.saveas(local_output_path)
            logger.info(f"✓ Composed DXF saved locally: {local_output_path}")
            
            # Upload to S3 if requested
            s3_url = None
            if upload_to_s3:
                logger.info("Uploading composed DXF to AWS S3...")
                s3_url = self.s3_manager.upload_file(
                    local_output_path,
                    output_filename,
                    content_type='application/dxf'
                )
                
                if s3_url:
                    logger.info(f"✓ File uploaded to S3: {s3_url}")
                else:
                    logger.warning("S3 upload failed, but local file is available")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare results
            results = {
                "success": True,
                "s3_url": s3_url,
                "local_path": local_output_path if not upload_to_s3 else None,
                "processing_time_seconds": round(processing_time, 2),
                "canvas_information": {
                    "width_inches": canvas.width_inches,
                    "height_inches": canvas.height_inches,
                    "thickness_inches": canvas.thickness_inches,
                    "has_overlaps": canvas.has_overlaps,
                    "width_mm": self._inches_to_mm(canvas.width_inches),
                    "height_mm": self._inches_to_mm(canvas.height_inches),
                    "thickness_mm": self._inches_to_mm(canvas.thickness_inches),
                    "volume_cubic_inches": round(
                        canvas.width_inches * canvas.height_inches * canvas.thickness_inches, 2
                    ),
                    "has_3d_representation": True
                },
                "layout_metadata": {
                    "layout_name": metadata.layout_name,
                    "brand": metadata.brand,
                    "container_type": metadata.container_type
                },
                "tools_placed": placed_tools,
                "total_tools": len(tools),
                "total_entities": sum(t["entities_count"] for t in placed_tools),
                "overlap_warnings": overlap_warnings,
                "file_size_bytes": os.path.getsize(local_output_path) if os.path.exists(local_output_path) else 0
            }
            
            logger.info("=" * 80)
            logger.info(f"✓ COMPOSITION COMPLETE in {processing_time:.2f}s")
            logger.info(f"✓ {len(placed_tools)} tools placed with {results['total_entities']} entities")
            if s3_url:
                logger.info(f"✓ S3 URL: {s3_url}")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Canvas composition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "s3_url": None,
                "local_path": None,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        finally:
            # Cleanup cache after processing
            self._cleanup_cache()
    
    def _apply_transformations(self, entity, rotation_degrees: float, flip_horizontal: bool, flip_vertical: bool):
        """Apply transformations to an entity"""
        try:
            import math
            
            # Apply rotation
            if abs(rotation_degrees) > 0.01:
                angle_rad = math.radians(rotation_degrees)
                if hasattr(entity, 'rotate_z'):
                    entity.rotate_z(angle_rad)
                else:
                    self._rotate_entity_manually(entity, angle_rad)
            
            # Apply flips
            scale_x = -1 if flip_horizontal else 1
            scale_y = -1 if flip_vertical else 1
            
            if scale_x != 1 or scale_y != 1:
                if hasattr(entity, 'scale'):
                    entity.scale(scale_x, scale_y, 1)
                else:
                    self._scale_entity_manually(entity, scale_x, scale_y)
                    
        except Exception as e:
            logger.warning(f"Failed to apply transformations to entity {type(entity)}: {e}")

    def _rotate_entity_manually(self, entity, angle_rad: float):
        """Manually rotate entity coordinates"""
        import math
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        def rotate_point(x, y):
            return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        
        try:
            if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                start_x, start_y = rotate_point(entity.dxf.start.x, entity.dxf.start.y)
                end_x, end_y = rotate_point(entity.dxf.end.x, entity.dxf.end.y)
                entity.dxf.start = (start_x, start_y, entity.dxf.start.z)
                entity.dxf.end = (end_x, end_y, entity.dxf.end.z)
                
            elif hasattr(entity.dxf, 'center'):
                center_x, center_y = rotate_point(entity.dxf.center.x, entity.dxf.center.y)
                entity.dxf.center = (center_x, center_y, entity.dxf.center.z)
                
        except Exception as e:
            logger.warning(f"Manual rotation failed for {type(entity)}: {e}")

    def _scale_entity_manually(self, entity, scale_x: float, scale_y: float):
        """Manually scale entity coordinates"""
        try:
            if hasattr(entity.dxf, 'start') and hasattr(entity.dxf, 'end'):
                entity.dxf.start = (entity.dxf.start.x * scale_x, entity.dxf.start.y * scale_y, entity.dxf.start.z)
                entity.dxf.end = (entity.dxf.end.x * scale_x, entity.dxf.end.y * scale_y, entity.dxf.end.z)
                
            elif hasattr(entity.dxf, 'center'):
                entity.dxf.center = (entity.dxf.center.x * scale_x, entity.dxf.center.y * scale_y, entity.dxf.center.z)
                if hasattr(entity.dxf, 'radius'):
                    entity.dxf.radius = entity.dxf.radius * abs(scale_x)
                    
        except Exception as e:
            logger.warning(f"Manual scaling failed for {type(entity)}: {e}")


# ============================================================================
# API ENDPOINT FUNCTIONS FOR HUGGING FACE DEPLOYMENT
# ============================================================================

def create_canvas_dxf_api(request_data: Dict) -> Dict:
    """
    Main API endpoint for DXF canvas composition with AWS S3 integration
    
    Expected request_data format:
    {
        "canvas_information": {
            "width_inches": 21,
            "height_inches": 11,
            "thickness_inches": 0.5,
            "has_overlaps": false
        },
        "layout_metadata": {
            "layout_name": "My Layout",
            "brand": "Makita",
            "container_type": "Drawer"
        },
        "tools": [
            {
                "tool_id": "unique-id",
                "name": "Tool Name",
                "brand": "Brand",
                "dxf_link": "https://s3-url-to-dxf-file",
                "position_inches": {"x": 2.0, "y": 1.0},
                "rotation_degrees": 0,
                "height_diagonal_inches": 5.0,
                "thickness_inches": 0.5,
                "flip_horizontal": false,
                "flip_vertical": false,
                "opacity": 100,
                "smooth": 0
            }
        ],
        "output_filename": "final_combined.dxf",
        "upload_to_s3": true
    }
    
    Returns:
    {
        "success": true/false,
        "s3_url": "https://s3-url-to-composed-dxf" or null,
        "error": "error message" if failed,
        "processing_time_seconds": 12.34,
        "canvas_information": {...},
        "tools_placed": [...],
        ...
    }
    """
    try:
        # Validate required fields
        if "canvas_information" not in request_data:
            return {
                "success": False,
                "error": "canvas_information is required",
                "s3_url": None
            }
        
        if "tools" not in request_data or not request_data["tools"]:
            return {
                "success": False,
                "error": "tools array is required and must not be empty",
                "s3_url": None
            }
        
        # Validate all tools have DXF links
        for i, tool in enumerate(request_data["tools"]):
            if "dxf_link" not in tool or not tool["dxf_link"]:
                return {
                    "success": False,
                    "error": f"Tool at index {i} is missing dxf_link",
                    "s3_url": None
                }
        
        # Extract optional parameters
        output_filename = request_data.get("output_filename", None)
        upload_to_s3 = request_data.get("upload_to_s3", True)
        
        # Initialize composer
        logger.info("Initializing DXF Canvas Composer...")
        composer = DXFCanvasComposer()
        
        # Generate canvas
        results = composer.compose_canvas_from_json(
            layout_json=request_data,
            output_filename=output_filename,
            upload_to_s3=upload_to_s3
        )
        
        return results
        
    except Exception as e:
        logger.error(f"API endpoint error: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"API endpoint error: {str(e)}",
            "s3_url": None
        }


def health_check() -> Dict:
    """Health check endpoint for monitoring"""
    try:
        s3_manager = S3Manager()
        s3_available = s3_manager.s3_client is not None
        
        return {
            "status": "healthy",
            "s3_available": s3_available,
            "cache_dir": HF_CACHE_DIR,
            "cache_exists": os.path.exists(HF_CACHE_DIR),
            "aws_region": AWS_REGION,
            "s3_bucket": S3_BUCKET_NAME
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_api_with_mock_data():
    """
    Test the API with mock data (for development)
    """
    
    # Mock request data
    request_data = {
        "canvas_information": {
            "width_inches": 21,
            "height_inches": 11,
            "thickness_inches": 0.5,
            "has_overlaps": False
        },
        "layout_metadata": {
            "layout_name": "Test_Layout_3D",
            "brand": "Makita",
            "container_type": "Drawer"
        },
        "tools": [
            {
                "tool_id": "tool-001",
                "name": "Wrench",
                "brand": "Makita",
                "dxf_link": "https://axionbucketmlb.s3.eu-north-1.amazonaws.com/wrench.dxf",
                "position_inches": {"x": 2, "y": 1},
                "rotation_degrees": 0,
                "height_diagonal_inches": 8.5,
                "thickness_inches": 0.5,
                "flip_horizontal": False,
                "flip_vertical": False,
                "opacity": 100,
                "smooth": 0
            },
            {
                "tool_id": "tool-002",
                "name": "Pliers",
                "brand": "Milwaukee",
                "dxf_link": "https://axionbucketmlb.s3.eu-north-1.amazonaws.com/pliers.dxf",
                "position_inches": {"x": 10, "y": 1},
                "rotation_degrees": 45,
                "height_diagonal_inches": 7.0,
                "thickness_inches": 0.5,
                "flip_horizontal": False,
                "flip_vertical": False,
                "opacity": 100,
                "smooth": 0
            }
        ],
        "output_filename": "test_composed_canvas_3d.dxf",
        "upload_to_s3": True
    }
    
    # Call API
    print("=" * 80)
    print("TESTING DXF CANVAS COMPOSER API")
    print("=" * 80)
    
    results = create_canvas_dxf_api(request_data)
    
    # Print results
    print("\n" + "=" * 80)
    print("API RESPONSE:")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    
    if results["success"]:
        print("\n✓ SUCCESS!")
        if results.get("s3_url"):
            print(f"✓ S3 URL: {results['s3_url']}")
        print(f"✓ Processing time: {results['processing_time_seconds']}s")
        print(f"✓ Tools placed: {results['total_tools']}")
        print(f"✓ Total entities: {results['total_entities']}")
        print(f"✓ Canvas dimensions: {results['canvas_information']['width_inches']}\" x {results['canvas_information']['height_inches']}\" x {results['canvas_information']['thickness_inches']}\"")
        print(f"✓ 3D Canvas: {results['canvas_information']['has_3d_representation']}")
    else:
        print(f"\n❌ FAILED: {results['error']}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run health check
    print("Running health check...")
    health = health_check()
    print(json.dumps(health, indent=2))
    print()
    
    # Run test (only if you have actual DXF files at the URLs)
    # test_api_with_mock_data()