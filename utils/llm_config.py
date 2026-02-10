"""
LLM Configuration for Llama 3.3 70B
Handles API integration with Groq
"""

import os
from typing import Optional

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not available. Install: pip install langchain-groq")


class LLMConfig:
    """Manages LLM instances for the application"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM configuration
        
        Args:
            api_key: Groq API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.llm = None
        
        if GROQ_AVAILABLE and self.api_key:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Llama 3.3 70B model"""
        try:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=self.api_key,
                temperature=0.7,  # Balanced creativity
                max_tokens=1024
            )
            print("✅ Llama 3.3 70B initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            self.llm = None
    
    def get_llm(self):
        """Get the LLM instance"""
        return self.llm
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.llm is not None


# Singleton instance
_llm_config = None

def get_llm_instance(api_key: Optional[str] = None):
    """
    Get or create LLM instance
    
    Args:
        api_key: Optional API key
        
    Returns:
        LLM instance or None
    """
    global _llm_config
    
    if _llm_config is None:
        _llm_config = LLMConfig(api_key)
    
    return _llm_config.get_llm()


if __name__ == "__main__":
    # Test the LLM
    llm = get_llm_instance()
    if llm:
        print("Testing LLM...")
        response = llm.invoke("Say hello!")
        print(f"Response: {response.content}")