"""
AI Chat Service for K2 Quant

Supports OpenAI GPT models and Anthropic Claude with streaming responses
for collaborative strategy development.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
from dataclasses import dataclass, asdict

import openai
import anthropic
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic

from k2_quant.utilities.logger import k2_logger


@dataclass
class Message:
    """Chat message structure"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API calls"""
        return {
            'role': self.role,
            'content': self.content
        }


class AIProvider:
    """Base class for AI providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.conversation_history: List[Message] = []
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        message = Message(role, content, metadata=metadata)
        self.conversation_history.append(message)
        return message
    
    def get_history_for_api(self, max_messages: int = 20) -> List[Dict]:
        """Get conversation history formatted for API"""
        recent_messages = self.conversation_history[-max_messages:]
        return [msg.to_dict() for msg in recent_messages]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider with streaming support"""
    
    AVAILABLE_MODELS = [
        'gpt-4-turbo-preview',
        'gpt-4',
        'gpt-4-32k',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k'
    ]
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.current_model = 'gpt-4'
    
    def set_model(self, model_name: str):
        """Set the model to use"""
        if model_name in self.AVAILABLE_MODELS:
            self.current_model = model_name
            k2_logger.info(f"OpenAI model set to: {model_name}", "AI_CHAT")
        else:
            k2_logger.warning(f"Model {model_name} not available", "AI_CHAT")
    
    def get_streaming_response(self, message: str, system_prompt: str = None) -> Generator:
        """Get streaming response from OpenAI"""
        # Add user message to history
        self.add_message('user', message)
        
        # Prepare messages for API
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        # Add conversation history
        messages.extend(self.get_history_for_api())
        
        try:
            # Create streaming completion
            stream = self.client.chat.completions.create(
                model=self.current_model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=2000
            )
            
            full_response = ""
            
            # Stream the response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Add assistant response to history
            self.add_message('assistant', full_response)
            
        except Exception as e:
            error_msg = f"OpenAI API error: {str(e)}"
            k2_logger.error(error_msg, "AI_CHAT")
            yield f"\n[Error: {error_msg}]"
    
    async def get_async_response(self, message: str, system_prompt: str = None) -> str:
        """Get async response from OpenAI"""
        # Add user message to history
        self.add_message('user', message)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.extend(self.get_history_for_api())
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.current_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            self.add_message('assistant', content)
            return content
            
        except Exception as e:
            error_msg = f"OpenAI API error: {str(e)}"
            k2_logger.error(error_msg, "AI_CHAT")
            return f"Error: {error_msg}"
    
    def generate_strategy_code(self, description: str, requirements: List[str]) -> Dict[str, Any]:
        """Generate Python code for a trading strategy"""
        system_prompt = """You are an expert quantitative analyst and Python programmer.
        Generate clean, efficient Python code for stock market analysis and projections.
        The code should work with pandas DataFrames and numpy arrays.
        Include detailed comments explaining the mathematical logic.
        """
        
        user_prompt = f"""Create a Python function for the following trading strategy:
        
        Description: {description}
        
        Requirements:
        {chr(10).join(f'- {req}' for req in requirements)}
        
        The function should:
        1. Accept a pandas DataFrame with columns: date_time, open, high, low, close, volume
        2. Perform the calculations as described
        3. Return the DataFrame with new columns added for projections/signals
        4. Include error handling for edge cases
        
        Provide the code in a format ready to execute.
        """
        
        # Get response
        response_text = ""
        for chunk in self.get_streaming_response(user_prompt, system_prompt):
            response_text += chunk
        
        # Extract code from response
        code = self.extract_code_from_response(response_text)
        
        return {
            'success': True,
            'code': code,
            'description': description,
            'full_response': response_text
        }
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from AI response"""
        # Look for code blocks
        if '```python' in response:
            parts = response.split('```python')
            if len(parts) > 1:
                code_part = parts[1].split('```')[0]
                return code_part.strip()
        elif '```' in response:
            parts = response.split('```')
            if len(parts) > 1:
                code_part = parts[1].split('```')[0]
                return code_part.strip()
        
        # Return full response if no code blocks found
        return response


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider with streaming support"""
    
    AVAILABLE_MODELS = [
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-2.1',
        'claude-2.0'
    ]
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        self.current_model = 'claude-3-opus-20240229'
    
    def set_model(self, model_name: str):
        """Set the model to use"""
        if model_name in self.AVAILABLE_MODELS:
            self.current_model = model_name
            k2_logger.info(f"Anthropic model set to: {model_name}", "AI_CHAT")
    
    def get_streaming_response(self, message: str, system_prompt: str = None) -> Generator:
        """Get streaming response from Anthropic"""
        # Add user message to history
        self.add_message('user', message)
        
        # Prepare prompt
        prompt = ""
        if system_prompt:
            prompt = f"{system_prompt}\n\n"
        
        # Add conversation history
        for msg in self.get_history_for_api():
            if msg['role'] == 'user':
                prompt += f"Human: {msg['content']}\n\n"
            elif msg['role'] == 'assistant':
                prompt += f"Assistant: {msg['content']}\n\n"
        
        prompt += "Assistant: "
        
        try:
            # Create streaming completion
            stream = self.client.completions.create(
                model=self.current_model,
                prompt=prompt,
                max_tokens_to_sample=2000,
                stream=True,
                temperature=0.7
            )
            
            full_response = ""
            
            # Stream the response
            for chunk in stream:
                if hasattr(chunk, 'completion'):
                    content = chunk.completion
                    full_response += content
                    yield content
            
            # Add assistant response to history
            self.add_message('assistant', full_response)
            
        except Exception as e:
            error_msg = f"Anthropic API error: {str(e)}"
            k2_logger.error(error_msg, "AI_CHAT")
            yield f"\n[Error: {error_msg}]"


class AIChatService:
    """Main AI chat service managing multiple providers"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider = None
        self.system_context = None
        
        # Initialize providers if API keys are available
        self.initialize_providers()
    
    def initialize_providers(self):
        """Initialize available AI providers"""
        # OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.providers['openai'] = OpenAIProvider(openai_key)
            self.current_provider = 'openai'
            k2_logger.info("OpenAI provider initialized", "AI_CHAT")
        
        # Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.providers['anthropic'] = AnthropicProvider(anthropic_key)
            if not self.current_provider:
                self.current_provider = 'anthropic'
            k2_logger.info("Anthropic provider initialized", "AI_CHAT")
        
        if not self.providers:
            k2_logger.warning("No AI providers available - add API keys to .env", "AI_CHAT")
    
    def set_provider(self, provider_name: str):
        """Switch AI provider"""
        if provider_name in self.providers:
            self.current_provider = provider_name
            k2_logger.info(f"Switched to {provider_name} provider", "AI_CHAT")
            return True
        return False
    
    def set_model(self, model_name: str):
        """Set model for current provider"""
        if self.current_provider and self.current_provider in self.providers:
            self.providers[self.current_provider].set_model(model_name)
    
    def set_system_context(self, context: str):
        """Set system context for strategy development"""
        self.system_context = f"""You are an expert quantitative analyst helping develop trading strategies.
        You have deep knowledge of:
        - Mathematical formulas and statistical analysis
        - Technical indicators and chart patterns
        - Python programming with pandas, numpy, and scipy
        - Time series analysis and forecasting
        
        Context: {context}
        
        When developing strategies:
        1. Ask clarifying questions to understand requirements fully
        2. Break down complex calculations into clear steps
        3. Provide detailed mathematical explanations
        4. Generate clean, well-commented Python code
        5. Focus on accuracy and performance for large datasets
        """
    
    def get_streaming_response(self, message: str) -> Generator:
        """Get streaming response from current provider"""
        if not self.current_provider or self.current_provider not in self.providers:
            yield "No AI provider available. Please configure API keys."
            return
        
        provider = self.providers[self.current_provider]
        
        # Use system context if available
        system_prompt = self.system_context or None
        
        # Stream response
        for chunk in provider.get_streaming_response(message, system_prompt):
            yield chunk
    
    def generate_strategy_code(self, description: str, requirements: List[str]) -> Dict[str, Any]:
        """Generate strategy code using current provider"""
        if not self.current_provider or self.current_provider not in self.providers:
            return {
                'success': False,
                'error': 'No AI provider available'
            }
        
        provider = self.providers[self.current_provider]
        
        if isinstance(provider, OpenAIProvider):
            return provider.generate_strategy_code(description, requirements)
        else:
            # Fallback for other providers
            prompt = f"Create a Python trading strategy: {description}"
            response = ""
            for chunk in provider.get_streaming_response(prompt):
                response += chunk
            
            return {
                'success': True,
                'code': response,
                'description': description
            }
    
    def clear_history(self):
        """Clear conversation history for all providers"""
        for provider in self.providers.values():
            provider.clear_history()
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for each provider"""
        models = {}
        
        if 'openai' in self.providers:
            models['openai'] = OpenAIProvider.AVAILABLE_MODELS
        
        if 'anthropic' in self.providers:
            models['anthropic'] = AnthropicProvider.AVAILABLE_MODELS
        
        return models
    
    def save_conversation(self, filepath: str):
        """Save conversation history to file"""
        if self.current_provider and self.current_provider in self.providers:
            provider = self.providers[self.current_provider]
            
            conversation = {
                'provider': self.current_provider,
                'timestamp': datetime.now().isoformat(),
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat()
                    }
                    for msg in provider.conversation_history
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(conversation, f, indent=2)
            
            k2_logger.info(f"Conversation saved to {filepath}", "AI_CHAT")
    
    def load_conversation(self, filepath: str):
        """Load conversation history from file"""
        try:
            with open(filepath, 'r') as f:
                conversation = json.load(f)
            
            provider_name = conversation.get('provider')
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                provider.clear_history()
                
                for msg_data in conversation.get('messages', []):
                    provider.add_message(
                        msg_data['role'],
                        msg_data['content']
                    )
                
                k2_logger.info(f"Conversation loaded from {filepath}", "AI_CHAT")
                return True
                
        except Exception as e:
            k2_logger.error(f"Failed to load conversation: {str(e)}", "AI_CHAT")
            return False


# Singleton instance
ai_chat_service = AIChatService()