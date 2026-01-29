"""
Cloud Integration for Chromatica Pro

Remote processing, asset management, and collaboration features.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import threading
from queue import Queue
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CloudJob:
    """Cloud processing job."""
    job_id: str
    source_key: str
    reference_key: str
    strategy: str
    status: str  # pending, processing, completed, failed
    created: str
    completed: Optional[str] = None
    result_key: Optional[str] = None
    metrics: Optional[Dict] = None
    error: Optional[str] = None


class CloudStorage:
    """
    Abstract cloud storage interface.
    
    Implementations: S3, GCS, Azure Blob, local filesystem
    """
    
    def upload(self, local_path: str, remote_key: str) -> bool:
        """Upload file to cloud."""
        raise NotImplementedError
    
    def download(self, remote_key: str, local_path: str) -> bool:
        """Download file from cloud."""
        raise NotImplementedError
    
    def exists(self, remote_key: str) -> bool:
        """Check if remote key exists."""
        raise NotImplementedError
    
    def delete(self, remote_key: str) -> bool:
        """Delete remote file."""
        raise NotImplementedError
    
    def list_keys(self, prefix: str) -> List[str]:
        """List keys with prefix."""
        raise NotImplementedError


class LocalCloudStorage(CloudStorage):
    """Local filesystem implementation of cloud storage."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload(self, local_path: str, remote_key: str) -> bool:
        import shutil
        dest = self.base_path / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return True
    
    def download(self, remote_key: str, local_path: str) -> bool:
        import shutil
        source = self.base_path / remote_key
        if source.exists():
            shutil.copy2(source, local_path)
            return True
        return False
    
    def exists(self, remote_key: str) -> bool:
        return (self.base_path / remote_key).exists()
    
    def delete(self, remote_key: str) -> bool:
        path = self.base_path / remote_key
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_keys(self, prefix: str) -> List[str]:
        prefix_path = self.base_path / prefix
        if prefix_path.exists():
            return [str(p.relative_to(self.base_path)) 
                   for p in prefix_path.rglob("*") if p.is_file()]
        return []


class CloudProcessor:
    """
    Cloud processing orchestrator.
    
    Manages job queue, asset upload/download, and result retrieval.
    """
    
    def __init__(self, storage: CloudStorage, 
                 config: Optional[Dict] = None):
        self.storage = storage
        self.config = config or {}
        
        self.jobs: Dict[str, CloudJob] = {}
        self.job_queue = Queue()
        
        self._worker_running = False
        self._worker_thread = None
    
    def submit_job(self, source_path: str, reference_path: str,
                   strategy: str = 'auto') -> str:
        """
        Submit a processing job.
        
        Args:
            source_path: Local source image path
            reference_path: Local reference image path
            strategy: Matching strategy
            
        Returns:
            Job ID
        """
        # Generate job ID
        job_id = hashlib.md5(
            f"{source_path}{reference_path}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Upload assets
        source_key = f"jobs/{job_id}/source{Path(source_path).suffix}"
        reference_key = f"jobs/{job_id}/reference{Path(reference_path).suffix}"
        
        self.storage.upload(source_path, source_key)
        self.storage.upload(reference_path, reference_key)
        
        # Create job
        job = CloudJob(
            job_id=job_id,
            source_key=source_key,
            reference_key=reference_key,
            strategy=strategy,
            status='pending',
            created=datetime.now().isoformat()
        )
        
        self.jobs[job_id] = job
        self.job_queue.put(job_id)
        
        logger.info(f"Submitted cloud job: {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status."""
        job = self.jobs.get(job_id)
        if job:
            return {
                'job_id': job.job_id,
                'status': job.status,
                'created': job.created,
                'completed': job.completed,
                'metrics': job.metrics,
                'error': job.error
            }
        return None
    
    def get_result(self, job_id: str, output_path: str) -> bool:
        """Download job result."""
        job = self.jobs.get(job_id)
        if job and job.status == 'completed' and job.result_key:
            return self.storage.download(job.result_key, output_path)
        return False
    
    def start_worker(self):
        """Start background processing worker."""
        if self._worker_running:
            return
        
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self._worker_thread.start()
        logger.info("Cloud worker started")
    
    def stop_worker(self):
        """Stop background worker."""
        self._worker_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
    
    def _process_jobs(self):
        """Background job processing loop."""
        import tempfile
        
        while self._worker_running:
            try:
                job_id = self.job_queue.get(timeout=1)
            except:
                continue
            
            job = self.jobs.get(job_id)
            if not job:
                continue
            
            logger.info(f"Processing job: {job_id}")
            job.status = 'processing'
            
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    
                    # Download assets
                    source_local = tmpdir / "source.tmp"
                    reference_local = tmpdir / "reference.tmp"
                    result_local = tmpdir / "result.tmp"
                    
                    self.storage.download(job.source_key, str(source_local))
                    self.storage.download(job.reference_key, str(reference_local))
                    
                    # Process
                    from ..engines.hybrid.orchestrator import HybridOrchestrator
                    from ..io.formats import ImageIO
                    
                    source, _ = ImageIO.read(str(source_local))
                    reference, _ = ImageIO.read(str(reference_local))
                    
                    orchestrator = HybridOrchestrator()
                    result = orchestrator.match(source, reference, strategy=job.strategy)
                    
                    # Save and upload result
                    ImageIO.write(str(result_local), result['output'])
                    
                    result_key = f"jobs/{job_id}/result.png"
                    self.storage.upload(str(result_local), result_key)
                    
                    # Update job
                    job.status = 'completed'
                    job.completed = datetime.now().isoformat()
                    job.result_key = result_key
                    job.metrics = result.get('metrics', {})
                    
                    logger.info(f"Job completed: {job_id}")
                    
            except Exception as e:
                job.status = 'failed'
                job.error = str(e)
                logger.error(f"Job failed: {job_id} - {e}")
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict]:
        """List all jobs, optionally filtered by status."""
        jobs = []
        for job in self.jobs.values():
            if status is None or job.status == status:
                jobs.append(self.get_job_status(job.job_id))
        return jobs
    
    def cleanup_job(self, job_id: str):
        """Delete job and associated files."""
        job = self.jobs.get(job_id)
        if job:
            self.storage.delete(job.source_key)
            self.storage.delete(job.reference_key)
            if job.result_key:
                self.storage.delete(job.result_key)
            del self.jobs[job_id]
            logger.info(f"Cleaned up job: {job_id}")
