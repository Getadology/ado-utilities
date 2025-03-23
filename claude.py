//
//  Config.swift
//  ado-utilities
//
//  Created by bill donner on 3/18/25.
//


"""
AWS Python Advertising Content Processing System

A scalable system that acquires remote advertising content from various sources,
processes it with external AI routines, and efficiently stores the results in
both vector and relational databases.
"""
import os
import sys
import json
import asyncio
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import aiohttp
import boto3
import aioboto3
import faiss
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("adcontent-processor")

# Configuration
@dataclass
class Config:
    """System configuration parameters"""
    # AWS settings
    aws_region: str = "us-east-1"
    s3_bucket: str = "ad-content-processing"
    sqs_queue_url: str = "https://sqs.us-east-1.amazonaws.com/123456789012/ad-content-queue"
    
    # Database settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "adcontent"
    db_user: str = "postgres"
    db_password: str = "postgres"
    
    # Processing settings
    max_concurrent_downloads: int = 10
    max_concurrent_ai_tasks: int = 5
    temp_dir: str = "/tmp/adcontent"
    
    # FAISS settings
    vector_dimension: int = 512
    faiss_index_path: str = "/data/faiss/ad_content_index"

    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        return cls(
            aws_region=os.environ.get("AWS_REGION", cls.aws_region),
            s3_bucket=os.environ.get("S3_BUCKET", cls.s3_bucket),
            sqs_queue_url=os.environ.get("SQS_QUEUE_URL", cls.sqs_queue_url),
            db_host=os.environ.get("DB_HOST", cls.db_host),
            db_port=int(os.environ.get("DB_PORT", cls.db_port)),
            db_name=os.environ.get("DB_NAME", cls.db_name),
            db_user=os.environ.get("DB_USER", cls.db_user),
            db_password=os.environ.get("DB_PASSWORD", cls.db_password),
            max_concurrent_downloads=int(os.environ.get("MAX_CONCURRENT_DOWNLOADS", cls.max_concurrent_downloads)),
            max_concurrent_ai_tasks=int(os.environ.get("MAX_CONCURRENT_AI_TASKS", cls.max_concurrent_ai_tasks)),
            temp_dir=os.environ.get("TEMP_DIR", cls.temp_dir),
            vector_dimension=int(os.environ.get("VECTOR_DIMENSION", cls.vector_dimension)),
            faiss_index_path=os.environ.get("FAISS_INDEX_PATH", cls.faiss_index_path),
        )


# Data Models
@dataclass
class ContentItem:
    """Represents a single advertising content item"""
    url: str
    brand_id: str
    content_type: str  # "image" or "video"
    content_id: str = None
    local_path: Optional[str] = None
    s3_key: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.content_id:
            self.content_id = str(uuid.uuid4())
        if not self.metadata:
            self.metadata = {}


@dataclass
class AIAnalysisResult:
    """Results from AI analysis of content"""
    content_id: str
    timestamp: datetime = None
    embedding: Optional[List[float]] = None
    features: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()
        if not self.features:
            self.features = {}
        if not self.metrics:
            self.metrics = {}
        if not self.tags:
            self.tags = []


@dataclass
class AggregatedAnalysis:
    """Aggregated analysis results for a brand"""
    brand_id: str
    timestamp: datetime = None
    content_ids: List[str] = None
    brand_metrics: Dict[str, float] = None
    brand_tags: List[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()
        if not self.content_ids:
            self.content_ids = []
        if not self.brand_metrics:
            self.brand_metrics = {}
        if not self.brand_tags:
            self.brand_tags = []


@dataclass
class ProcessingJob:
    """A complete processing job for a brand's content"""
    job_id: str
    brand_id: str
    urls: List[str]
    status: str = "pending"  # pending, processing, completed, failed
    start_time: datetime = None
    end_time: datetime = None
    content_items: List[ContentItem] = None
    individual_results: Dict[str, AIAnalysisResult] = None
    aggregated_result: Optional[AggregatedAnalysis] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now()
        if not self.content_items:
            self.content_items = []
        if not self.individual_results:
            self.individual_results = {}


# Content Acquisition Module
class ContentAcquisitionService:
    """Handles downloading content from URLs and uploading to S3"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.s3_client = None
        
    async def setup(self):
        """Initialize resources"""
        self.session = aiohttp.ClientSession()
        self.s3_client = aioboto3.Session().client('s3', region_name=self.config.aws_region)
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
    async def cleanup(self):
        """Release resources"""
        if self.session:
            await self.session.close()
        if self.s3_client:
            await self.s3_client.close()
    
    async def download_content(self, content_item: ContentItem) -> ContentItem:
        """Download content from URL to local storage"""
        try:
            logger.info(f"Downloading content from {content_item.url}")
            
            # Determine file extension based on headers or URL
            async with self.session.head(content_item.url) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    content_item.content_type = 'image'
                    ext = self._get_image_extension(content_type)
                elif 'video' in content_type:
                    content_item.content_type = 'video'
                    ext = self._get_video_extension(content_type)
                else:
                    # Try to determine from URL
                    url_parts = content_item.url.split('.')
                    if len(url_parts) > 1:
                        ext = url_parts[-1].lower()
                        if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                            content_item.content_type = 'image'
                        elif ext in ['mp4', 'webm', 'avi', 'mov']:
                            content_item.content_type = 'video'
                    else:
                        ext = 'bin'  # Default unknown extension
                
            # Create local file path
            local_filename = f"{content_item.content_id}.{ext}"
            local_path = os.path.join(self.config.temp_dir, local_filename)
            content_item.local_path = local_path
            
            # Download the file
            async with self.session.get(content_item.url) as response:
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            
            # Upload to S3
            s3_key = f"{content_item.brand_id}/{content_item.content_id}.{ext}"
            content_item.s3_key = s3_key
            
            async with self.s3_client as s3:
                await s3.upload_file(
                    local_path, 
                    self.config.s3_bucket, 
                    s3_key,
                    ExtraArgs={'ContentType': content_type or f"{content_item.content_type}/{ext}"}
                )
            
            logger.info(f"Successfully downloaded and uploaded {content_item.url} to S3 at {s3_key}")
            return content_item
            
        except Exception as e:
            logger.error(f"Error downloading content from {content_item.url}: {str(e)}")
            raise
    
    def _get_image_extension(self, content_type: str) -> str:
        """Determine image file extension from content type"""
        if 'jpeg' in content_type or 'jpg' in content_type:
            return 'jpg'
        elif 'png' in content_type:
            return 'png'
        elif 'gif' in content_type:
            return 'gif'
        elif 'webp' in content_type:
            return 'webp'
        else:
            return 'jpg'  # Default to jpg
    
    def _get_video_extension(self, content_type: str) -> str:
        """Determine video file extension from content type"""
        if 'mp4' in content_type:
            return 'mp4'
        elif 'webm' in content_type:
            return 'webm'
        elif 'quicktime' in content_type:
            return 'mov'
        else:
            return 'mp4'  # Default to mp4


# AI Processing Interface
class AIMethod:
    """Base class for AI processing methods"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    async def process(self, content_item: ContentItem) -> Dict[str, Any]:
        """Process a content item and return analysis results"""
        raise NotImplementedError("Subclasses must implement process()")


class ImageFeatureExtractor(AIMethod):
    """Extract features from image content"""
    
    def __init__(self):
        super().__init__(
            name="image_feature_extractor",
            description="Extracts visual features and embeddings from images"
        )
    
    async def process(self, content_item: ContentItem) -> Dict[str, Any]:
        """
        Extract features from an image using external AI service
        In a real implementation, this would call Amazon Rekognition,
        Google Cloud Vision, or a similar service
        """
        if content_item.content_type != "image":
            raise ValueError("This processor only works with image content")
            
        # Simulate API call to external service
        logger.info(f"Extracting features from image: {content_item.content_id}")
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Mock response - in reality would come from AI service
        return {
            "embedding": list(np.random.rand(512).astype(float)),  # Random embedding vector
            "features": {
                "dominant_colors": ["#336699", "#FFFFFF", "#222222"],
                "brightness": 0.75,
                "contrast": 0.65,
                "complexity": 0.55
            },
            "metrics": {
                "quality_score": 0.82,
                "appeal_score": 0.75
            },
            "tags": ["outdoor", "product", "lifestyle"]
        }


class VideoAnalyzer(AIMethod):
    """Analyze video content"""
    
    def __init__(self):
        super().__init__(
            name="video_analyzer",
            description="Analyzes video content for features, key frames, and embeddings"
        )
    
    async def process(self, content_item: ContentItem) -> Dict[str, Any]:
        """
        Analyze video using external AI service
        In a real implementation, this would call a video processing API
        """
        if content_item.content_type != "video":
            raise ValueError("This processor only works with video content")
            
        # Simulate API call to external service
        logger.info(f"Analyzing video: {content_item.content_id}")
        await asyncio.sleep(1.0)  # Videos take longer to process
        
        # Mock response - in reality would come from AI service
        return {
            "embedding": list(np.random.rand(512).astype(float)),  # Random embedding vector
            "features": {
                "duration": 15.3,
                "frame_rate": 30,
                "key_frames": [0.0, 3.5, 7.2, 12.8],
                "scene_changes": [0.0, 5.2, 10.1]
            },
            "metrics": {
                "engagement_score": 0.78,
                "quality_score": 0.85,
                "motion_score": 0.62
            },
            "tags": ["promotional", "product_demo", "fast_paced"]
        }


class BrandAnalyzer(AIMethod):
    """Analyze aggregated content for a brand"""
    
    def __init__(self):
        super().__init__(
            name="brand_analyzer",
            description="Analyzes aggregated content for consistent brand identity"
        )
    
    async def process_aggregated(self, 
                               brand_id: str, 
                               individual_results: Dict[str, AIAnalysisResult]) -> Dict[str, Any]:
        """Process aggregated results for a brand"""
        logger.info(f"Performing aggregated brand analysis for {brand_id}")
        
        # In a real implementation, this would perform more sophisticated analysis
        # across all content items to identify brand patterns and consistency
        
        # Get all tags across items
        all_tags = []
        for result in individual_results.values():
            all_tags.extend(result.tags)
        
        # Count tag frequency
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Get top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        top_tags = [tag for tag, count in top_tags[:5]]
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in ["quality_score", "appeal_score", "engagement_score"]:
            values = []
            for result in individual_results.values():
                if result.metrics and metric_name in result.metrics:
                    values.append(result.metrics[metric_name])
            if values:
                avg_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
        
        # Calculate consistency score based on embedding similarity
        consistency_score = self._calculate_consistency_score([
            result.embedding for result in individual_results.values() if result.embedding
        ])
        
        return {
            "brand_metrics": {
                **avg_metrics,
                "consistency_score": consistency_score,
                "content_variety": min(1.0, len(individual_results) / 10),  # Higher with more content
            },
            "brand_tags": top_tags
        }
    
    def _calculate_consistency_score(self, embeddings: List[List[float]]) -> float:
        """
        Calculate a consistency score based on embedding similarity
        Higher score means more consistent brand identity
        """
        if len(embeddings) < 2:
            return 0.5  # Default for not enough data
            
        # Convert to numpy arrays
        np_embeddings = np.array(embeddings)
        
        # Calculate pairwise cosine similarities
        norms = np.linalg.norm(np_embeddings, axis=1, keepdims=True)
        normalized = np_embeddings / norms
        similarities = np.dot(normalized, normalized.T)
        
        # Remove self-similarities (diagonal)
        np.fill_diagonal(similarities, 0)
        
        # Average similarity
        avg_similarity = np.sum(similarities) / (similarities.size - len(embeddings))
        
        # Scale to 0-1 range (typical cosine similarities might range from 0.5 to 0.95)
        consistency_score = (avg_similarity - 0.5) * 2
        return max(0, min(1, consistency_score))


# AI Module registry and factory
class AIModuleRegistry:
    """Registry for available AI processing modules"""
    
    def __init__(self):
        self.modules = {}
    
    def register(self, method_class):
        """Register an AI method class"""
        instance = method_class()
        self.modules[instance.name] = instance
        return instance
    
    def get_method(self, name: str) -> AIMethod:
        """Get an AI method by name"""
        if name not in self.modules:
            raise ValueError(f"AI method '{name}' not registered")
        return self.modules[name]
    
    def list_methods(self) -> List[Dict[str, str]]:
        """List all available methods"""
        return [
            {"name": method.name, "description": method.description}
            for method in self.modules.values()
        ]


# Database Services
class DatabaseService:
    """Handle database operations for storing processing results"""
    
    def __init__(self, config: Config):
        self.config = config
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL database"""
        self.conn = psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password
        )
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def store_job_results(self, job: ProcessingJob):
        """Store processing job results in the relational database"""
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                # Insert job record
                cur.execute(
                    """
                    INSERT INTO processing_jobs
                    (job_id, brand_id, status, start_time, end_time, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    end_time = EXCLUDED.end_time,
                    error_message = EXCLUDED.error_message
                    """,
                    (
                        job.job_id, 
                        job.brand_id, 
                        job.status, 
                        job.start_time, 
                        job.end_time,
                        job.error_message
                    )
                )
                
                # Insert content items
                if job.content_items:
                    content_items_data = [
                        (
                            item.content_id,
                            job.job_id,
                            item.brand_id,
                            item.url,
                            item.content_type,
                            item.s3_key,
                            json.dumps(item.metadata) if item.metadata else None
                        )
                        for item in job.content_items
                    ]
                    
                    execute_values(
                        cur,
                        """
                        INSERT INTO content_items
                        (content_id, job_id, brand_id, url, content_type, s3_key, metadata)
                        VALUES %s
                        ON CONFLICT (content_id) DO UPDATE SET
                        s3_key = EXCLUDED.s3_key,
                        metadata = EXCLUDED.metadata
                        """,
                        content_items_data
                    )
                
                # Insert individual analysis results
                if job.individual_results:
                    analysis_data = [
                        (
                            str(uuid.uuid4()),  # analysis_id
                            result.content_id,
                            job.job_id,
                            result.timestamp,
                            json.dumps(result.features) if result.features else None,
                            json.dumps(result.metrics) if result.metrics else None,
                            json.dumps(result.tags) if result.tags else None
                        )
                        for result in job.individual_results.values()
                    ]
                    
                    execute_values(
                        cur,
                        """
                        INSERT INTO content_analysis
                        (analysis_id, content_id, job_id, timestamp, features, metrics, tags)
                        VALUES %s
                        """,
                        analysis_data
                    )
                
                # Insert aggregated brand analysis
                if job.aggregated_result:
                    cur.execute(
                        """
                        INSERT INTO brand_analysis
                        (brand_id, job_id, timestamp, metrics, tags)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (brand_id, job_id) DO UPDATE SET
                        timestamp = EXCLUDED.timestamp,
                        metrics = EXCLUDED.metrics,
                        tags = EXCLUDED.tags
                        """,
                        (
                            job.brand_id,
                            job.job_id,
                            job.aggregated_result.timestamp,
                            json.dumps(job.aggregated_result.brand_metrics) 
                                if job.aggregated_result.brand_metrics else None,
                            json.dumps(job.aggregated_result.brand_tags)
                                if job.aggregated_result.brand_tags else None
                        )
                    )
                
                conn.commit()
                logger.info(f"Successfully stored results for job {job.job_id} in database")
        
        except Exception as e:
            logger.error(f"Error storing job results in database: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            self.close()


class VectorDBService:
    """Handle vector database operations using FAISS"""
    
    def __init__(self, config: Config):
        self.config = config
        self.index = None
        self.content_ids = []  # To map index positions to content IDs
    
    def initialize(self):
        """Initialize FAISS index"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.faiss_index_path), exist_ok=True)
            
            # Try to load existing index
            if os.path.exists(self.config.faiss_index_path):
                self.index = faiss.read_index(self.config.faiss_index_path)
                # Load content ID mapping
                mapping_file = f"{self.config.faiss_index_path}.ids"
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        self.content_ids = json.load(f)
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.config.vector_dimension)
                self.content_ids = []
                
            logger.info(f"FAISS index initialized with {len(self.content_ids)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            return False
    
    def store_embeddings(self, job: ProcessingJob):
        """Store embeddings from analysis results in FAISS index"""
        try:
            if not self.index:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize FAISS index")
            
            # Collect embeddings and content IDs
            new_embeddings = []
            new_content_ids = []
            
            for content_id, result in job.individual_results.items():
                if result.embedding and len(result.embedding) == self.config.vector_dimension:
                    new_embeddings.append(result.embedding)
                    new_content_ids.append(content_id)
            
            if new_embeddings:
                # Convert to numpy array and add to index
                embeddings_array = np.array(new_embeddings).astype('float32')
                self.index.add(embeddings_array)
                
                # Update content ID mapping
                self.content_ids.extend(new_content_ids)
                
                # Save index and mapping
                faiss.write_index(self.index, self.config.faiss_index_path)
                with open(f"{self.config.faiss_index_path}.ids", 'w') as f:
                    json.dump(self.content_ids, f)
                
                logger.info(f"Added {len(new_embeddings)} vectors to FAISS index")
            else:
                logger.info("No valid embeddings to add to FAISS index")
                
        except Exception as e:
            logger.error(f"Error storing embeddings in FAISS index: {str(e)}")
            raise
    
    def search_similar(self, embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar content using embedding vector"""
        if not self.index:
            if not self.initialize():
                raise RuntimeError("Failed to initialize FAISS index")
        
        # Convert query to numpy array
        query = np.array([embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Map results to content IDs
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.content_ids):
                results.append((self.content_ids[idx], float(distances[0][i])))
        
        return results


# Processing Pipeline
class ContentProcessingPipeline:
    """Main processing pipeline for content analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.acquisition_service = ContentAcquisitionService(config)
        self.db_service = DatabaseService(config)
        self.vector_db_service = VectorDBService(config)
        
        # Register AI methods
        self.ai_registry = AIModuleRegistry()
        self.image_processor = self.ai_registry.register(ImageFeatureExtractor)
        self.video_processor = self.ai_registry.register(VideoAnalyzer)
        self.brand_analyzer = self.ai_registry.register(BrandAnalyzer)
    
    async def setup(self):
        """Initialize services"""
        await self.acquisition_service.setup()
        self.vector_db_service.initialize()
    
    async def cleanup(self):
        """Clean up resources"""
        await self.acquisition_service.cleanup()
    
    async def process_job(self, job: ProcessingJob) -> ProcessingJob:
        """Process a complete job for a brand's content"""
        try:
            logger.info(f"Starting processing job {job.job_id} for brand {job.brand_id}")
            job.status = "processing"
            
            # Step 1: Create ContentItem objects for each URL
            for url in job.urls:
                content_item = ContentItem(url=url, brand_id=job.brand_id)
                job.content_items.append(content_item)
            
            # Step 2: Acquire content (download and upload to S3)
            semaphore = asyncio.Semaphore(self.config.max_concurrent_downloads)
            
            async def download_with_semaphore(item):
                async with semaphore:
                    return await self.acquisition_service.download_content(item)
            
            download_tasks = [
                download_with_semaphore(item) for item in job.content_items
            ]
            job.content_items = await asyncio.gather(*download_tasks)
            
            # Step 3: Process individual content items with AI methods
            ai_semaphore = asyncio.Semaphore(self.config.max_concurrent_ai_tasks)
            
            async def process_content_item(item):
                async with ai_semaphore:
                    logger.info(f"Processing content item {item.content_id}")
                    
                    # Select appropriate processor based on content type
                    processor = self.image_processor if item.content_type == "image" else self.video_processor
                    
                    # Process the item
                    result_dict = await processor.process(item)
                    
                    # Create and return analysis result
                    return AIAnalysisResult(
                        content_id=item.content_id,
                        embedding=result_dict.get("embedding"),
                        features=result_dict.get("features"),
                        metrics=result_dict.get("metrics"),
                        tags=result_dict.get("tags")
                    )
            
            # Process all content items concurrently
            processing_tasks = [
                process_content_item(item) for item in job.content_items
            ]
            individual_results = await asyncio.gather(*processing_tasks)
            
            # Store results by content ID
            job.individual_results = {
                result.content_id: result for result in individual_results
            }
            
            # Step 4: Perform aggregated brand analysis
            aggregated_results = await self.brand_analyzer.process_aggregated(
                job.brand_id, job.individual_results
            )
            
            job.aggregated_result = AggregatedAnalysis(
                brand_id=job.brand_id,
                content_ids=[item.content_id for item in job.content_items],
                brand_metrics=aggregated_results.get("brand_metrics"),
                brand_tags=aggregated_results.get("brand_tags")
            )
            
            # Step 5: Store results in databases
            # Store embeddings in vector database
            self.vector_db_service.store_embeddings(job)
            
            # Store metadata and analysis in relational database
            job.end_time = datetime.now()
            job.status = "completed"
            self.db_service.store_job_results(job)
            
            logger.info(f"Successfully completed processing job {job.job_id}")
            return job
            
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now()
            job.error_message = str(e)
            logger.error(f"Error processing job {job.job_id}: {str(e)}")
            
            # Try to save failed job status
            try:
                self.db_service.store_job_results(job)
            except Exception as db_error:
                logger.error(f"Failed to store error state in database: {str(db_error)}")
                
            raise


# Job Input Handlers
class CommandLineHandler:
    """Handle job input from command line arguments"""
    
    @staticmethod
    def parse_args():
                 """Parse command line arguments"""
 import argparse
                                    parser = argparse.ArgumentParser(description="Process advertising content")
                                    parser.add_argument("--url-file", required=True, help="File containing URLs to process")
                                    parser.add_argument("--brand-id", required=True, help="Brand ID associated with the content")
                                    parser.add_argument("--job-id", help="Job ID (optional, will be generated if not provided)")
                                    return parser.parse_args()
                                
                                @staticmethod
                                def read_urls_from_file(file_path):
                                    """Read URLs from a file"""
                                    with open(file_path, 'r') as f:
                                        return [line.strip() for line in f if line.strip()]
                                
                                @staticmethod
                                def create_job_from_args():
                                    """Create a processing job from command line arguments"""
                                    args = CommandLineHandler.parse_args()
                                    
                                    # Read URLs from file
                                    urls = CommandLineHandler.read_urls_from_file(args.url_file)
                                    
                                    # Create job
                                    job_id = args.job_id if args.job_id else str(uuid.uuid4())
                                    return ProcessingJob(
                                        job_id=job_id,
                                        brand_id=args.brand_id,
                                        urls=urls
                                    )


                            class SQSHandler:
                                """Handle job input from SQS queue"""
                                
                                def __init__(self, config: Config):
                                    self.config = config
                                    self.sqs_client = boto3.client('sqs', region_name=config.aws_region)
                                
                                def receive_messages(self, max_messages=10):
                                    """Receive messages from SQS queue"""
                                    response = self.sqs_client.receive_message(
                                        QueueUrl=self.config.sqs_queue_url,
                                        MaxNumberOfMessages=max_messages,
                                        WaitTimeSeconds=20,  # Long polling
                                        VisibilityTimeout=300  # 5 minutes
                                    )
                                    
                                    return response.get('Messages', [])
                                
                                def delete_message(self, receipt_handle):
                                    """Delete a message from the queue"""
                                    self.sqs_client.delete_message(
                                        QueueUrl=self.config.sqs_queue_url,
                                        ReceiptHandle=receipt_handle
                                    )
                                
                                def create_job_from_message(self, message):
                                    """Create a processing job from an SQS message"""
                                    try:
                                        # Parse message body
                                        body = json.loads(message['Body'])
                                        
                                        # Extract job details
                                        job_id = body.get('job_id', str(uuid.uuid4()))
                                        brand_id = body.get('brand_id')
                                        urls = body.get('urls', [])
                                        
                                        if not brand_id:
                                            raise ValueError("SQS message must contain a brand_id")
                                        
                                        if not urls:
                                            raise ValueError("SQS message must contain a list of URLs")
                                        
                                        # Create job
                                        return ProcessingJob(
                                            job_id=job_id,
                                            brand_id=brand_id,
                                            urls=urls
                                        )
                                        
                                    except Exception as e:
                                        logger.error(f"Error parsing SQS message: {str(e)}")
                                        raise


                            # Testing Harness
                            class TestingHarness:
                                """Testing framework for individual AI methods"""
                                
                                def __init__(self, config: Config):
                                    self.config = config
                                    self.ai_registry = AIModuleRegistry()
                                    self.image_processor = self.ai_registry.register(ImageFeatureExtractor)
                                    self.video_processor = self.ai_registry.register(VideoAnalyzer)
                                    self.brand_analyzer = self.ai_registry.register(BrandAnalyzer)
                                
                                def list_available_methods(self):
                                    """List available AI methods"""
                                    return self.ai_registry.list_methods()
                                
                                async def test_image_processor(self, image_path, content_id=None):
                                    """Test image processor with a local image file"""
                                    if not os.path.exists(image_path):
                                        raise FileNotFoundError(f"Image file not found: {image_path}")
                                    
                                    # Create a content item
                                    content_item = ContentItem(
                                        url=f"file://{image_path}",
                                        brand_id="test_brand",
                                        content_type="image",
                                        content_id=content_id or str(uuid.uuid4()),
                                        local_path=image_path
                                    )
                                    
                                    # Process the item
                                    result = await self.image_processor.process(content_item)
                                    
                                    # Return the result
                                    return {
                                        "content_id": content_item.content_id,
                                        "result": result
                                    }
                                
                                async def test_video_processor(self, video_path, content_id=None):
                                    """Test video processor with a local video file"""
                                    if not os.path.exists(video_path):
                                        raise FileNotFoundError(f"Video file not found: {video_path}")
                                    
                                    # Create a content item
                                    content_item = ContentItem(
                                        url=f"file://{video_path}",
                                        brand_id="test_brand",
                                        content_type="video",
                                        content_id=content_id or str(uuid.uuid4()),
                                        local_path=video_path
                                    )
                                    
                                    # Process the item
                                    result = await self.video_processor.process(content_item)
                                    
                                    # Return the result
                                    return {
                                        "content_id": content_item.content_id,
                                        "result": result
                                    }
                                
                                async def test_brand_analyzer(self, analysis_results):
                                    """Test brand analyzer with a set of analysis results"""
                                    # Convert to AIAnalysisResult objects if necessary
                                    individual_results = {}
                                    for content_id, result in analysis_results.items():
                                        if isinstance(result, dict):
                                            individual_results[content_id] = AIAnalysisResult(
                                                content_id=content_id,
                                                embedding=result.get("embedding"),
                                                features=result.get("features"),
                                                metrics=result.get("metrics"),
                                                tags=result.get("tags")
                                            )
                                        else:
                                            individual_results[content_id] = result
                                    
                                    # Process the aggregated results
                                    result = await self.brand_analyzer.process_aggregated("test_brand", individual_results)
                                    
                                    # Return the result
                                    return {
                                        "brand_id": "test_brand",
                                        "result": result
                                    }


                            # Main Application
                            class AdContentProcessor:
                                """Main application class"""
                                
                                def __init__(self):
                                    self.config = Config.from_env()
                                    self.pipeline = ContentProcessingPipeline(self.config)
                                    self.sqs_handler = SQSHandler(self.config)
                                    self.testing_harness = TestingHarness(self.config)
                                
                                async def initialize(self):
                                    """Initialize the application"""
                                    await self.pipeline.setup()
                                
                                async def shutdown(self):
                                    """Clean up resources"""
                                    await self.pipeline.cleanup()
                                
                                async def process_command_line(self):
                                    """Process job from command line arguments"""
                                    job = CommandLineHandler.create_job_from_args()
                                    await self.pipeline.process_job(job)
                                    return job
                                
                                async def process_sqs_queue(self, max_messages=10):
                                    """Process jobs from SQS queue"""
                                    messages = self.sqs_handler.receive_messages(max_messages)
                                    jobs = []
                                    
                                    for message in messages:
                                        try:
                                            # Create job from message
                                            job = self.sqs_handler.create_job_from_message(message)
                                            
                                            # Process the job
                                            await self.pipeline.process_job(job)
                                            
                                            # Delete the message
                                            self.sqs_handler.delete_message(message['ReceiptHandle'])
                                            
                                            jobs.append(job)
                                            
                                        except Exception as e:
                                            logger.error(f"Error processing SQS message: {str(e)}")
                                    
                                    return jobs
                                
                                async def run_tests(self, test_config):
                                    """Run tests using the testing harness"""
                                    results = {}
                                    
                                    # Test image processor
                                    if "image_paths" in test_config:
                                        image_results = []
                                        for image_path in test_config["image_paths"]:
                                            result = await self.testing_harness.test_image_processor(image_path)
                                            image_results.append(result)
                                        results["image_processor"] = image_results
                                    
                                    # Test video processor
                                    if "video_paths" in test_config:
                                        video_results = []
                                        for video_path in test_config["video_paths"]:
                                            result = await self.testing_harness.test_video_processor(video_path)
                                            video_results.append(result)
                                        results["video_processor"] = video_results
                                    
                                    # Test brand analyzer
                                    if "analysis_results" in test_config:
                                        brand_result = await self.testing_harness.test_brand_analyzer(
                                            test_config["analysis_results"]
                                        )
                                        results["brand_analyzer"] = brand_result
                                    
                                    return results


                            # Database setup script
                            def setup_database():
                                """Create database tables if they don't exist"""
                                config = Config.from_env()
                                db_service = DatabaseService(config)
                                
                                try:
                                    conn = db_service.connect()
                                    with conn.cursor() as cur:
                                        # Create processing_jobs table
                                        cur.execute("""
                                            CREATE TABLE IF NOT EXISTS processing_jobs (
                                                job_id VARCHAR(36) PRIMARY KEY,
                                                brand_id VARCHAR(36) NOT NULL,
                                                status VARCHAR(20) NOT NULL,
                                                start_time TIMESTAMP NOT NULL,
                                                end_time TIMESTAMP,
                                                error_message TEXT
                                            )
                                        """)
                                        
                                        # Create content_items table
                                        cur.execute("""
                                            CREATE TABLE IF NOT EXISTS content_items (
                                                content_id VARCHAR(36) PRIMARY KEY,
                                                job_id VARCHAR(36) NOT NULL,
                                                brand_id VARCHAR(36) NOT NULL,
                                                url TEXT NOT NULL,
                                                content_type VARCHAR(20) NOT NULL,
                                                s3_key TEXT,
                                                metadata JSONB,
                                                FOREIGN KEY (job_id) REFERENCES processing_jobs(job_id)
                                            )
                                        """)
                                        
                                        # Create content_analysis table
                                        cur.execute("""
                                            CREATE TABLE IF NOT EXISTS content_analysis (
                                                analysis_id VARCHAR(36) PRIMARY KEY,
                                                content_id VARCHAR(36) NOT NULL,
                                                job_id VARCHAR(36) NOT NULL,
                                                timestamp TIMESTAMP NOT NULL,
                                                features JSONB,
                                                metrics JSONB,
                                                tags JSONB,
                                                FOREIGN KEY (content_id) REFERENCES content_items(content_id),
                                                FOREIGN KEY (job_id) REFERENCES processing_jobs(job_id)
                                            )
                                        """)
                                        
                                        # Create brand_analysis table
                                        cur.execute("""
                                            CREATE TABLE IF NOT EXISTS brand_analysis (
                                                brand_id VARCHAR(36) NOT NULL,
                                                job_id VARCHAR(36) NOT NULL,
                                                timestamp TIMESTAMP NOT NULL,
                                                metrics JSONB,
                                                tags JSONB,
                                                PRIMARY KEY (brand_id, job_id),
                                                FOREIGN KEY (job_id) REFERENCES processing_jobs(job_id)
                                            )
                                        """)
                                        
                                        conn.commit()
                                        logger.info("Database tables created successfully")
                                except Exception as e:
                                    logger.error(f"Error setting up database: {str(e)}")
                                    if conn:
                                        conn.rollback()
                                finally:
                                    db_service.close()


                            # Entry point
                            async def main():
                                """Main entry point"""
                                parser = argparse.ArgumentParser(description="Ad Content Processing System")
                                parser.add_argument("--setup-db", action="store_true", help="Set up database tables")
                                parser.add_argument("--mode", choices=["cli", "sqs", "test"], default="cli",
                                                  help="Operation mode: cli, sqs, or test")
                                parser.add_argument("--test-config", help="Path to test configuration file")
                                args = parser.parse_args()
                                
                                # Set up database if requested
                                if args.setup_db:
                                    setup_database()
                                    return
                                
                                # Initialize processor
                                processor = AdContentProcessor()
                                await processor.initialize()
                                
                                try:
                                    if args.mode == "cli":
                                        # Process from command line
                                        job = await processor.process_command_line()
                                        print(f"Job {job.job_id} completed with status {job.status}")
                                        
                                    elif args.mode == "sqs":
                                        # Process from SQS
                                        jobs = await processor.process_sqs_queue()
                                        print(f"Processed {len(jobs)} jobs from SQS")
                                        
                                    elif args.mode == "test":
                                        # Run tests
                                        if not args.test_config:
                                            print("Error: --test-config is required for test mode")
                                            return
                                            
                                        with open(args.test_config, 'r') as f:
                                            test_config = json.load(f)
                                            
                                        results = await processor.run_tests(test_config)
                                        print(json.dumps(results, indent=2))
                                
                                finally:
                                    # Clean up
                                    await processor.shutdown()


                            if __name__ == "__main__":
                                asyncio.run(main())
