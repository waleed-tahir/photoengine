"""
Workflow Management

Project, session, and history management for professional workflows.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchHistoryEntry:
    """Single match operation history entry."""
    timestamp: str
    source_path: str
    reference_path: str
    output_path: Optional[str]
    strategy: str
    delta_e_mean: float
    execution_time: float
    parameters_file: Optional[str]


@dataclass
class Project:
    """Project container for color matching workflow."""
    name: str
    created: str
    modified: str
    source_folder: str
    reference_folder: str
    output_folder: str
    default_strategy: str
    history: List[MatchHistoryEntry]
    settings: Dict


class WorkflowManager:
    """
    Manage color matching projects and sessions.
    
    Features:
    - Project creation and loading
    - Match history tracking
    - Auto-save/restore session state
    - Parameter presets
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        if workspace_dir:
            self.workspace = Path(workspace_dir)
        else:
            self.workspace = Path.home() / ".chromatica"
        
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.projects_dir = self.workspace / "projects"
        self.presets_dir = self.workspace / "presets"
        self.sessions_dir = self.workspace / "sessions"
        
        for d in [self.projects_dir, self.presets_dir, self.sessions_dir]:
            d.mkdir(exist_ok=True)
        
        self.current_project: Optional[Project] = None
    
    def create_project(self, name: str, 
                       source_folder: str = "",
                       reference_folder: str = "",
                       output_folder: str = "") -> Project:
        """Create a new project."""
        now = datetime.now().isoformat()
        
        project = Project(
            name=name,
            created=now,
            modified=now,
            source_folder=source_folder,
            reference_folder=reference_folder,
            output_folder=output_folder,
            default_strategy='auto',
            history=[],
            settings={}
        )
        
        self.save_project(project)
        self.current_project = project
        
        logger.info(f"Created project: {name}")
        return project
    
    def save_project(self, project: Project):
        """Save project to disk."""
        project.modified = datetime.now().isoformat()
        
        project_file = self.projects_dir / f"{project.name}.json"
        
        # Convert to dict
        data = {
            'name': project.name,
            'created': project.created,
            'modified': project.modified,
            'source_folder': project.source_folder,
            'reference_folder': project.reference_folder,
            'output_folder': project.output_folder,
            'default_strategy': project.default_strategy,
            'history': [asdict(h) for h in project.history],
            'settings': project.settings
        }
        
        with open(project_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_project(self, name: str) -> Project:
        """Load project from disk."""
        project_file = self.projects_dir / f"{name}.json"
        
        if not project_file.exists():
            raise FileNotFoundError(f"Project not found: {name}")
        
        with open(project_file, 'r') as f:
            data = json.load(f)
        
        history = [MatchHistoryEntry(**h) for h in data.get('history', [])]
        
        project = Project(
            name=data['name'],
            created=data['created'],
            modified=data['modified'],
            source_folder=data.get('source_folder', ''),
            reference_folder=data.get('reference_folder', ''),
            output_folder=data.get('output_folder', ''),
            default_strategy=data.get('default_strategy', 'auto'),
            history=history,
            settings=data.get('settings', {})
        )
        
        self.current_project = project
        logger.info(f"Loaded project: {name}")
        return project
    
    def list_projects(self) -> List[str]:
        """List all available projects."""
        return [f.stem for f in self.projects_dir.glob("*.json")]
    
    def add_history(self, entry: MatchHistoryEntry):
        """Add entry to current project history."""
        if self.current_project:
            self.current_project.history.append(entry)
            self.save_project(self.current_project)
    
    def save_preset(self, name: str, parameters: dict):
        """Save parameter preset."""
        preset_file = self.presets_dir / f"{name}.json"
        with open(preset_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        logger.info(f"Saved preset: {name}")
    
    def load_preset(self, name: str) -> dict:
        """Load parameter preset."""
        preset_file = self.presets_dir / f"{name}.json"
        with open(preset_file, 'r') as f:
            return json.load(f)
    
    def list_presets(self) -> List[str]:
        """List available presets."""
        return [f.stem for f in self.presets_dir.glob("*.json")]
    
    def save_session(self, session_data: dict):
        """Save current session state for recovery."""
        session_file = self.sessions_dir / "last_session.json"
        session_data['timestamp'] = datetime.now().isoformat()
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def restore_session(self) -> Optional[dict]:
        """Restore last session."""
        session_file = self.sessions_dir / "last_session.json"
        
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return None
    
    def export_project(self, project_name: str, export_path: str):
        """Export project with all presets and history."""
        project = self.load_project(project_name)
        
        export_data = {
            'project': asdict(project) if hasattr(project, '__dict__') else {
                'name': project.name,
                'history': [asdict(h) for h in project.history],
                'settings': project.settings
            },
            'presets': {},
            'exported': datetime.now().isoformat()
        }
        
        # Include related presets
        for preset_name in self.list_presets():
            if project_name.lower() in preset_name.lower():
                export_data['presets'][preset_name] = self.load_preset(preset_name)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported project to: {export_path}")
