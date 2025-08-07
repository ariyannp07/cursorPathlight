"""
AI Assistant Module
Handles user queries and AI interactions
"""

import logging
import openai
from typing import Dict, Any, Optional


class AIAssistant:
    """AI assistant for handling user queries"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the AI assistant"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.provider = config.get('provider', 'openai')
        self.model = config.get('model', 'gpt-4')
        self.api_key = config.get('api_key', '')
        self.max_tokens = config.get('max_tokens', 150)
        self.temperature = config.get('temperature', 0.7)
        self.system_prompt = config.get('system_prompt', 'You are Pathlight, an AI assistant helping blind individuals navigate and socialize.')
        
        # Initialize AI provider
        self._initialize_ai()
        
    def _initialize_ai(self):
        """Initialize AI provider"""
        try:
            if self.provider == 'openai' and self.api_key:
                openai.api_key = self.api_key
                self.logger.info(f"AI assistant initialized: {self.provider} - {self.model}")
            else:
                self.logger.warning("AI assistant not fully configured - API key missing")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize AI assistant: {e}")
    
    def process_query(self, query: str) -> str:
        """
        Process user query and return response
        
        Args:
            query: User's question or request
            
        Returns:
            AI response
        """
        try:
            if not self.api_key:
                return "I'm sorry, but I'm not configured to answer questions right now. Please check my setup."
            
            # Process with OpenAI
            if self.provider == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                return response.choices[0].message.content
            
            return "I'm sorry, but I don't know how to process that request."
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return "I'm sorry, but I encountered an error while processing your request."
    
    def get_status(self) -> Dict[str, Any]:
        """Get AI assistant status"""
        return {
            'provider': self.provider,
            'model': self.model,
            'configured': bool(self.api_key),
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        } 