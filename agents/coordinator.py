import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import threading
import time

from .bana_architect import BanaArchitect
from .mussadiq_tester import MussadiqTester
from .mussahih_fixer import MussahihFixer


class AgentCoordinator:
    """Central coordinator for the multi-agent stock prediction system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.active_workflows = {}
        self.system_metrics = {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize agents
        self.initialize_agents()
        
        # Start coordinator services
        self.start_coordinator_services()
        
        self.logger.info("Agent Coordinator initialized successfully")
    
    def setup_logging(self):
        """Setup coordinator logging"""
        log_format = '[COORDINATOR] %(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/coordinator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Coordinator')
    
    def initialize_agents(self):
        """Initialize all three agents"""
        try:
            # Initialize Bana (Architect)
            self.agents['bana'] = BanaArchitect(self.config.get('bana', {}))
            
            # Initialize Mussadiq (Tester)  
            self.agents['mussadiq'] = MussadiqTester(self.config.get('mussadiq', {}))
            
            # Initialize Mussahih (Fixer)
            self.agents['mussahih'] = MussahihFixer(self.config.get('mussahih', {}))
            
            self.logger.info("All agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}")
            raise
    
    def start_coordinator_services(self):
        """Start background coordinator services"""
        # Start task processor
        self.task_processor_thread = threading.Thread(
            target=self.run_task_processor,
            daemon=True
        )
        self.task_processor_thread.start()
        
        # Schedule periodic tasks
        schedule.every(30).minutes.do(self.periodic_health_check)
        schedule.every(1).hours.do(self.generate_system_report)
        schedule.every().day.at("02:00").do(self.daily_maintenance)
        
        # Start scheduler
        self.scheduler_thread = threading.Thread(
            target=self.run_scheduler,
            daemon=True
        )
        self.scheduler_thread.start()
    
    def run_task_processor(self):
        """Run the main task processing loop"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.process_tasks())
    
    async def process_tasks(self):
        """Process tasks from the queue"""
        while True:
            try:
                task = await self.task_queue.get()
                await self.execute_workflow(task)
                self.task_queue.task_done()
            except Exception as e:
                self.logger.error(f"Task processing error: {str(e)}")
                await asyncio.sleep(1)
    
    def run_scheduler(self):
        """Run the periodic task scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    async def execute_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coordinated workflow between agents"""
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow_type = task.get('type')
        
        self.logger.info(f"Starting workflow {workflow_id}: {workflow_type}")
        
        workflow_result = {
            'workflow_id': workflow_id,
            'type': workflow_type,
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress',
            'steps': [],
            'result': {}
        }
        
        try:
            if workflow_type == 'full_development_cycle':
                workflow_result = await self.full_development_cycle(task, workflow_result)
            
            elif workflow_type == 'build_stock_model':
                workflow_result = await self.build_stock_model_workflow(task, workflow_result)
            
            elif workflow_type == 'test_and_fix_cycle':
                workflow_result = await self.test_and_fix_cycle(task, workflow_result)
            
            elif workflow_type == 'system_optimization':
                workflow_result = await self.system_optimization_workflow(task, workflow_result)
            
            elif workflow_type == 'emergency_response':
                workflow_result = await self.emergency_response_workflow(task, workflow_result)
            
            else:
                workflow_result['status'] = 'failed'
                workflow_result['error'] = f'Unknown workflow type: {workflow_type}'
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            workflow_result['status'] = 'failed'
            workflow_result['error'] = str(e)
        
        workflow_result['end_time'] = datetime.now().isoformat()
        workflow_result['duration'] = self.calculate_duration(
            workflow_result['start_time'], 
            workflow_result['end_time']
        )
        
        # Store workflow result
        self.active_workflows[workflow_id] = workflow_result
        
        return workflow_result
    
    async def full_development_cycle(self, task: Dict[str, Any], workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full development cycle: Build -> Test -> Fix -> Optimize"""
        
        # Step 1: Bana builds the system/feature
        build_task = {
            'type': 'create_desktop_component',
            'component_type': task.get('component_type', 'main_window'),
            'specifications': task.get('specifications', {})
        }
        
        bana_result = await self.agents['bana'].execute_task(build_task)
        workflow_result['steps'].append({
            'agent': 'bana',
            'action': 'build_component',
            'result': bana_result,
            'timestamp': datetime.now().isoformat()
        })
        
        if not bana_result.get('success', True):
            workflow_result['status'] = 'failed'
            return workflow_result
        
        # Step 2: Mussadiq tests the component
        test_task = {
            'type': 'run_full_test_suite'
        }
        
        mussadiq_result = await self.agents['mussadiq'].execute_task(test_task)
        workflow_result['steps'].append({
            'agent': 'mussadiq',
            'action': 'run_tests',
            'result': mussadiq_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 3: If tests fail, Mussahih analyzes and fixes
        if mussadiq_result.get('overall_summary', {}).get('success_rate', 100) < 95:
            # Analyze test failures
            analyze_task = {
                'type': 'analyze_test_failures',
                'test_results': mussadiq_result.get('test_suites', [])
            }
            
            mussahih_analysis = await self.agents['mussahih'].execute_task(analyze_task)
            workflow_result['steps'].append({
                'agent': 'mussahih',
                'action': 'analyze_failures',
                'result': mussahih_analysis,
                'timestamp': datetime.now().isoformat()
            })
            
            # Apply fixes
            if 'issue_id' in mussahih_analysis:
                fix_task = {
                    'type': 'fix_issue',
                    'issue_id': mussahih_analysis['issue_id'],
                    'auto_approve': task.get('auto_fix', False)
                }
                
                fix_result = await self.agents['mussahih'].execute_task(fix_task)
                workflow_result['steps'].append({
                    'agent': 'mussahih',
                    'action': 'apply_fixes',
                    'result': fix_result,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Step 4: Final optimization by Bana
        optimize_task = {
            'type': 'optimize_performance',
            'target_area': task.get('optimization_target')
        }
        
        optimization_result = await self.agents['bana'].execute_task(optimize_task)
        workflow_result['steps'].append({
            'agent': 'bana',
            'action': 'optimize_system',
            'result': optimization_result,
            'timestamp': datetime.now().isoformat()
        })
        
        workflow_result['status'] = 'completed'
        workflow_result['result'] = {
            'component_built': bana_result.get('success', False),
            'tests_passed': mussadiq_result.get('overall_summary', {}).get('success_rate', 0),
            'fixes_applied': len([step for step in workflow_result['steps'] if step['action'] == 'apply_fixes']),
            'optimization_completed': optimization_result.get('success', False)
        }
        
        return workflow_result
    
    async def build_stock_model_workflow(self, task: Dict[str, Any], workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Workflow to build stock prediction model with testing and validation"""
        
        symbol = task.get('symbol', 'AAPL')
        model_type = task.get('model_type', 'LSTM')
        
        # Step 1: Bana fetches stock data
        fetch_task = {
            'type': 'fetch_stock_data',
            'symbol': symbol,
            'period': task.get('period', '2y')
        }
        
        data_result = await self.agents['bana'].execute_task(fetch_task)
        workflow_result['steps'].append({
            'agent': 'bana',
            'action': 'fetch_data',
            'result': data_result,
            'timestamp': datetime.now().isoformat()
        })
        
        if not data_result.get('success', False):
            workflow_result['status'] = 'failed'
            return workflow_result
        
        # Step 2: Bana creates prediction model
        model_task = {
            'type': 'create_prediction_model',
            'model_type': model_type,
            'symbol': symbol,
            'parameters': task.get('model_parameters', {})
        }
        
        model_result = await self.agents['bana'].execute_task(model_task)
        workflow_result['steps'].append({
            'agent': 'bana',
            'action': 'create_model',
            'result': model_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 3: Bana generates predictions
        prediction_task = {
            'type': 'generate_predictions',
            'symbol': symbol,
            'model_type': model_type,
            'days': task.get('prediction_days', 30)
        }
        
        prediction_result = await self.agents['bana'].execute_task(prediction_task)
        workflow_result['steps'].append({
            'agent': 'bana',
            'action': 'generate_predictions',
            'result': prediction_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 4: Mussadiq validates predictions
        validation_task = {
            'type': 'validate_predictions',
            'symbol': symbol,
            'model_type': model_type
        }
        
        validation_result = await self.agents['mussadiq'].execute_task(validation_task)
        workflow_result['steps'].append({
            'agent': 'mussadiq',
            'action': 'validate_predictions',
            'result': validation_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 5: If validation shows issues, Mussahih diagnoses and fixes
        if validation_result.get('validation_status') == 'needs_review':
            diagnose_task = {
                'type': 'diagnose_issue',
                'issue_data': {
                    'description': f'Prediction accuracy issues for {symbol}',
                    'component': 'predictions',
                    'validation_result': validation_result
                }
            }
            
            diagnosis_result = await self.agents['mussahih'].execute_task(diagnose_task)
            workflow_result['steps'].append({
                'agent': 'mussahih',
                'action': 'diagnose_accuracy_issues',
                'result': diagnosis_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Apply recommended fixes
            if diagnosis_result.get('recommendations'):
                fix_task = {
                    'type': 'fix_issue',
                    'issue_id': diagnosis_result['issue_id'],
                    'auto_approve': task.get('auto_fix', True)
                }
                
                fix_result = await self.agents['mussahih'].execute_task(fix_task)
                workflow_result['steps'].append({
                    'agent': 'mussahih',
                    'action': 'improve_model',
                    'result': fix_result,
                    'timestamp': datetime.now().isoformat()
                })
        
        workflow_result['status'] = 'completed'
        workflow_result['result'] = {
            'symbol': symbol,
            'model_type': model_type,
            'data_records': data_result.get('records_count', 0),
            'predictions_generated': len(prediction_result.get('predictions', [])),
            'validation_status': validation_result.get('validation_status'),
            'accuracy_metrics': validation_result.get('accuracy_metrics', {})
        }
        
        return workflow_result
    
    async def test_and_fix_cycle(self, task: Dict[str, Any], workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Continuous test and fix cycle"""
        
        max_iterations = task.get('max_iterations', 3)
        current_iteration = 0
        
        while current_iteration < max_iterations:
            current_iteration += 1
            
            # Run tests
            test_task = {
                'type': 'run_full_test_suite'
            }
            
            test_result = await self.agents['mussadiq'].execute_task(test_task)
            workflow_result['steps'].append({
                'agent': 'mussadiq',
                'action': f'run_tests_iteration_{current_iteration}',
                'result': test_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check if tests pass
            success_rate = test_result.get('overall_summary', {}).get('success_rate', 0)
            target_success_rate = task.get('target_success_rate', 95)
            
            if success_rate >= target_success_rate:
                workflow_result['status'] = 'completed'
                workflow_result['result'] = {
                    'iterations_required': current_iteration,
                    'final_success_rate': success_rate,
                    'target_achieved': True
                }
                break
            
            # Analyze and fix issues
            analyze_task = {
                'type': 'analyze_test_failures',
                'test_results': test_result.get('test_suites', [])
            }
            
            analysis_result = await self.agents['mussahih'].execute_task(analyze_task)
            workflow_result['steps'].append({
                'agent': 'mussahih',
                'action': f'analyze_failures_iteration_{current_iteration}',
                'result': analysis_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Apply fixes
            if 'issue_id' in analysis_result:
                fix_task = {
                    'type': 'fix_issue',
                    'issue_id': analysis_result['issue_id'],
                    'auto_approve': True
                }
                
                fix_result = await self.agents['mussahih'].execute_task(fix_task)
                workflow_result['steps'].append({
                    'agent': 'mussahih',
                    'action': f'apply_fixes_iteration_{current_iteration}',
                    'result': fix_result,
                    'timestamp': datetime.now().isoformat()
                })
        
        # If we reach max iterations without success
        if workflow_result['status'] != 'completed':
            workflow_result['status'] = 'max_iterations_reached'
            workflow_result['result'] = {
                'iterations_completed': current_iteration,
                'final_success_rate': success_rate,
                'target_achieved': False,
                'requires_manual_intervention': True
            }
        
        return workflow_result
    
    async def emergency_response_workflow(self, task: Dict[str, Any], workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency response workflow for critical system issues"""
        
        # Immediate diagnosis
        diagnose_task = {
            'type': 'diagnose_issue',
            'issue_data': task.get('emergency_data', {})
        }
        
        diagnosis_result = await self.agents['mussahih'].execute_task(diagnose_task)
        workflow_result['steps'].append({
            'agent': 'mussahih',
            'action': 'emergency_diagnosis',
            'result': diagnosis_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Immediate fix application
        if diagnosis_result.get('recommendations'):
            fix_task = {
                'type': 'fix_issue',
                'issue_id': diagnosis_result['issue_id'],
                'auto_approve': True  # Emergency fixes are auto-approved
            }
            
            fix_result = await self.agents['mussahih'].execute_task(fix_task)
            workflow_result['steps'].append({
                'agent': 'mussahih',
                'action': 'emergency_fix',
                'result': fix_result,
                'timestamp': datetime.now().isoformat()
            })
        
        # System health check
        health_task = {
            'type': 'system_health_check'
        }
        
        health_result = await self.agents['bana'].execute_task(health_task)
        workflow_result['steps'].append({
            'agent': 'bana',
            'action': 'health_check',
            'result': health_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Validation tests
        validation_task = {
            'type': 'run_unit_tests',
            'module': 'critical'
        }
        
        validation_result = await self.agents['mussadiq'].execute_task(validation_task)
        workflow_result['steps'].append({
            'agent': 'mussadiq',
            'action': 'post_fix_validation',
            'result': validation_result,
            'timestamp': datetime.now().isoformat()
        })
        
        workflow_result['status'] = 'completed'
        workflow_result['result'] = {
            'emergency_resolved': health_result.get('overall_status') == 'healthy',
            'fixes_applied': len([s for s in workflow_result['steps'] if 'fix' in s['action']]),
            'system_stable': validation_result.get('summary', {}).get('success_rate', 0) > 90
        }
        
        return workflow_result
    
    def calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between timestamps in seconds"""
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        return (end - start).total_seconds()
    
    def periodic_health_check(self):
        """Periodic system health check"""
        self.logger.info("Running periodic health check")
        
        health_task = {
            'type': 'system_health_check'
        }
        
        asyncio.create_task(self.task_queue.put(health_task))
    
    def generate_system_report(self):
        """Generate comprehensive system report"""
        self.logger.info("Generating system report")
        
        report_task = {
            'type': 'generate_test_report',
            'format': 'html'
        }
        
        asyncio.create_task(self.task_queue.put(report_task))
    
    def daily_maintenance(self):
        """Daily maintenance routine"""
        self.logger.info("Running daily maintenance")
        
        maintenance_task = {
            'type': 'system_optimization'
        }
        
        asyncio.create_task(self.task_queue.put(maintenance_task))
    
    # Public API methods
    
    def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task to the coordinator queue"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task['task_id'] = task_id
        
        asyncio.create_task(self.task_queue.put(task))
        
        self.logger.info(f"Task {task_id} submitted: {task.get('type')}")
        return task_id
    
    def get_agent_status(self, agent_name: str = None) -> Dict[str, Any]:
        """Get status of specific agent or all agents"""
        if agent_name:
            if agent_name in self.agents:
                return self.agents[agent_name].get_status_report()
            else:
                return {'error': f'Agent {agent_name} not found'}
        
        return {
            'bana': self.agents['bana'].get_status_report(),
            'mussadiq': self.agents['mussadiq'].get_status_report(), 
            'mussahih': self.agents['mussahih'].get_status_report()
        }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of specific workflow"""
        return self.active_workflows.get(workflow_id, {'error': 'Workflow not found'})
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""
        return [
            {
                'workflow_id': wf_id,
                'type': wf['type'],
                'status': wf['status'],
                'start_time': wf['start_time']
            }
            for wf_id, wf in self.active_workflows.items()
            if wf['status'] == 'in_progress'
        ]
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        return {
            'timestamp': datetime.now().isoformat(),
            'agents': {
                'bana': {
                    'status': self.agents['bana'].status,
                    'capabilities': self.agents['bana'].get_capabilities()
                },
                'mussadiq': {
                    'status': self.agents['mussadiq'].status,
                    'capabilities': self.agents['mussadiq'].get_capabilities()
                },
                'mussahih': {
                    'status': self.agents['mussahih'].status,
                    'capabilities': self.agents['mussahih'].get_capabilities()
                }
            },
            'active_workflows': len([wf for wf in self.active_workflows.values() if wf['status'] == 'in_progress']),
            'total_workflows': len(self.active_workflows),
            'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
        } 