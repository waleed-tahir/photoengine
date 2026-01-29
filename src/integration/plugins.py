"""
Plugin System for Chromatica Pro

Extensible plugin architecture for custom transforms, exporters, and integrations.
"""

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: str
    enabled: bool = True


class PluginBase(ABC):
    """Base class for all plugins."""
    
    # Override in subclasses
    NAME = "base_plugin"
    VERSION = "1.0.0"
    AUTHOR = "Unknown"
    DESCRIPTION = "Base plugin"
    PLUGIN_TYPE = "generic"
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources when plugin is unloaded."""
        pass
    
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name=self.NAME,
            version=self.VERSION,
            author=self.AUTHOR,
            description=self.DESCRIPTION,
            plugin_type=self.PLUGIN_TYPE
        )


class TransformPlugin(PluginBase):
    """Plugin for custom color transforms."""
    
    PLUGIN_TYPE = "transform"
    
    @abstractmethod
    def apply(self, image, parameters: Dict) -> Any:
        """Apply the transform to an image."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """Get default parameters for the transform."""
        pass


class ExporterPlugin(PluginBase):
    """Plugin for custom export formats."""
    
    PLUGIN_TYPE = "exporter"
    
    @abstractmethod
    def export(self, parameters, output_path: str, options: Dict) -> bool:
        """Export parameters to custom format."""
        pass
    
    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension for this format."""
        pass


class IntegrationPlugin(PluginBase):
    """Plugin for external application integration."""
    
    PLUGIN_TYPE = "integration"
    
    @abstractmethod
    def connect(self, config: Dict) -> bool:
        """Connect to external application."""
        pass
    
    @abstractmethod
    def sync(self, data: Dict) -> Dict:
        """Sync data with external application."""
        pass


class PluginManager:
    """
    Manage plugins for Chromatica Pro.
    
    Features:
    - Automatic plugin discovery
    - Hot reloading
    - Dependency management
    - Enable/disable plugins
    
    Example:
        >>> manager = PluginManager()
        >>> manager.load_plugins('~/.chromatica/plugins')
        >>> transform = manager.get_plugin('my_transform', 'transform')
        >>> result = transform.apply(image, params)
    """
    
    def __init__(self, plugins_dir: Optional[str] = None):
        if plugins_dir:
            self.plugins_dir = Path(plugins_dir)
        else:
            self.plugins_dir = Path.home() / ".chromatica" / "plugins"
        
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, Dict[str, PluginBase]] = {
            'transform': {},
            'exporter': {},
            'integration': {},
            'generic': {}
        }
        
        self.plugin_info: Dict[str, PluginInfo] = {}
    
    def load_plugins(self, directory: Optional[str] = None):
        """Load all plugins from directory."""
        plugins_path = Path(directory) if directory else self.plugins_dir
        
        if not plugins_path.exists():
            logger.warning(f"Plugins directory not found: {plugins_path}")
            return
        
        for plugin_file in plugins_path.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                self._load_plugin_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
    
    def _load_plugin_file(self, plugin_path: Path):
        """Load a single plugin file."""
        spec = importlib.util.spec_from_file_location(
            plugin_path.stem, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PluginBase) and obj is not PluginBase:
                if obj not in [TransformPlugin, ExporterPlugin, IntegrationPlugin]:
                    self._register_plugin(obj)
    
    def _register_plugin(self, plugin_class: Type[PluginBase]):
        """Register a plugin class."""
        try:
            plugin = plugin_class()
            
            if plugin.initialize():
                info = plugin.get_info()
                plugin_type = info.plugin_type
                
                if plugin_type not in self.plugins:
                    plugin_type = 'generic'
                
                self.plugins[plugin_type][info.name] = plugin
                self.plugin_info[info.name] = info
                
                logger.info(f"Registered plugin: {info.name} ({info.plugin_type})")
            else:
                logger.warning(f"Plugin failed to initialize: {plugin_class.NAME}")
                
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
    
    def get_plugin(self, name: str, 
                   plugin_type: Optional[str] = None) -> Optional[PluginBase]:
        """Get a plugin by name and optionally type."""
        if plugin_type:
            return self.plugins.get(plugin_type, {}).get(name)
        
        for type_plugins in self.plugins.values():
            if name in type_plugins:
                return type_plugins[name]
        
        return None
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[PluginInfo]:
        """List all registered plugins."""
        if plugin_type:
            return [self.plugin_info[name] 
                   for name in self.plugins.get(plugin_type, {})]
        return list(self.plugin_info.values())
    
    def enable_plugin(self, name: str, enabled: bool = True):
        """Enable or disable a plugin."""
        if name in self.plugin_info:
            self.plugin_info[name].enabled = enabled
            logger.info(f"Plugin {name} {'enabled' if enabled else 'disabled'}")
    
    def unload_plugin(self, name: str):
        """Unload a plugin."""
        for type_plugins in self.plugins.values():
            if name in type_plugins:
                type_plugins[name].cleanup()
                del type_plugins[name]
                del self.plugin_info[name]
                logger.info(f"Unloaded plugin: {name}")
                return
    
    def reload_plugins(self):
        """Reload all plugins."""
        # Cleanup existing
        for type_plugins in self.plugins.values():
            for plugin in type_plugins.values():
                plugin.cleanup()
            type_plugins.clear()
        
        self.plugin_info.clear()
        
        # Reload
        self.load_plugins()


# Example built-in plugins

class IdentityTransform(TransformPlugin):
    """Example identity transform plugin."""
    
    NAME = "identity"
    VERSION = "1.0.0"
    AUTHOR = "Chromatica"
    DESCRIPTION = "Identity transform (no-op)"
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self):
        pass
    
    def apply(self, image, parameters: Dict):
        return image
    
    def get_parameters(self) -> Dict:
        return {}


class DCTLExporter(ExporterPlugin):
    """DaVinci DCTL exporter plugin."""
    
    NAME = "dctl_exporter"
    VERSION = "1.0.0"
    AUTHOR = "Chromatica"
    DESCRIPTION = "Export to DaVinci Resolve DCTL format"
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self):
        pass
    
    def export(self, parameters, output_path: str, options: Dict) -> bool:
        """Export to DCTL format."""
        # Generate DCTL code
        dctl_code = self._generate_dctl(parameters)
        
        with open(output_path, 'w') as f:
            f.write(dctl_code)
        
        return True
    
    def get_extension(self) -> str:
        return ".dctl"
    
    def _generate_dctl(self, parameters) -> str:
        """Generate DCTL shader code."""
        matrix = parameters.matrix
        
        return f'''// Generated by Chromatica Pro
// DaVinci DCTL Color Transform

DEFINE_UI_PARAMS(strength, Strength, DCTL_SLIDER_FLOAT, 1.0, 0.0, 1.0, 0.01)

__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)
{{
    // Color matrix
    float3x3 colorMatrix = make_float3x3(
        {matrix[0,0]:.6f}f, {matrix[0,1]:.6f}f, {matrix[0,2]:.6f}f,
        {matrix[1,0]:.6f}f, {matrix[1,1]:.6f}f, {matrix[1,2]:.6f}f,
        {matrix[2,0]:.6f}f, {matrix[2,1]:.6f}f, {matrix[2,2]:.6f}f
    );
    
    float3 offset = make_float3({matrix[0,3]:.6f}f, {matrix[1,3]:.6f}f, {matrix[2,3]:.6f}f);
    
    float3 rgb = make_float3(p_R, p_G, p_B);
    float3 result = mult_f3_f33(rgb, colorMatrix) + offset;
    
    // Blend with strength
    result = mix(rgb, result, strength);
    
    return result;
}}
'''
