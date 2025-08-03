import logging
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all agents in the Stock Price Projection system"""
    
    def __init__(self, name: str, role: str, config: Dict[str, Any] = None):
        self.name = name
        self.role = role
        self.config = config or {}
        self.status = "initialized"
        self.tasks = []
        self.results = []
        self.start_time = datetime.now()
        
        # Setup logging
        self.setup_logging()
        self.logger.info(f"Agent {self.name} ({self.role}) initialized")
        
        # Communication queue
        self.message_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        
    def setup_logging(self):
        """Setup agent-specific logging"""
        log_format = f'[{self.name}] %(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'logs/{self.name.lower()}_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.name)
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task - must be implemented by each agent"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass
    
    async def receive_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and process messages from other agents"""
        self.logger.info(f"Received message: {message}")
        await self.message_queue.put(message)
        
        # Process message based on type
        if message.get('type') == 'task':
            result = await self.execute_task(message['data'])
            return {
                'from': self.name,
                'to': message.get('from'),
                'type': 'task_result',
                'data': result,
                'timestamp': datetime.now().isoformat()
            }
        
        return {'status': 'acknowledged', 'agent': self.name}
    
    async def send_message(self, to_agent: str, message: Dict[str, Any]):
        """Send message to another agent"""
        message['from'] = self.name
        message['to'] = to_agent
        message['timestamp'] = datetime.now().isoformat()
        
        self.logger.info(f"Sending message to {to_agent}: {message}")
        await self.response_queue.put(message)
    
    def update_status(self, status: str, details: str = ""):
        """Update agent status"""
        self.status = status
        self.logger.info(f"Status updated to: {status} - {details}")
    
    def add_task(self, task: Dict[str, Any]):
        """Add task to agent's queue"""
        task['id'] = len(self.tasks) + 1
        task['status'] = 'pending'
        task['assigned_at'] = datetime.now().isoformat()
        self.tasks.append(task)
        self.logger.info(f"Task added: {task['description']}")
    
    def complete_task(self, task_id: int, result: Dict[str, Any]):
        """Mark task as complete and store result"""
        for task in self.tasks:
            if task['id'] == task_id:
                task['status'] = 'completed'
                task['completed_at'] = datetime.now().isoformat()
                break
        
        result['task_id'] = task_id
        result['completed_by'] = self.name
        result['timestamp'] = datetime.now().isoformat()
        self.results.append(result)
        
        self.logger.info(f"Task {task_id} completed")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        return {
            'agent': self.name,
            'role': self.role,
            'status': self.status,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'total_tasks': len(self.tasks),
            'completed_tasks': len([t for t in self.tasks if t['status'] == 'completed']),
            'pending_tasks': len([t for t in self.tasks if t['status'] == 'pending']),
            'capabilities': self.get_capabilities(),
            'last_activity': datetime.now().isoformat()
        }
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            'name': self.name,
            'role': self.role,
            'status': self.status,
            'tasks': self.tasks,
            'results': self.results,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load agent state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.tasks = state.get('tasks', [])
            self.results = state.get('results', [])
            self.config.update(state.get('config', {}))
            
            self.logger.info(f"State loaded from {filepath}")
        except FileNotFoundError:
            self.logger.warning(f"State file {filepath} not found")
    
    def __str__(self):
        return f"Agent({self.name}, {self.role}, {self.status})" 