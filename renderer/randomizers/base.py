from abc import ABC, abstractmethod

class BaseRandomizer(ABC):
    """Abstract base class for all randomizers."""
    
    @abstractmethod
    def apply(self, renderer, **kwargs):
        """Apply randomization to the renderer state."""
        pass
