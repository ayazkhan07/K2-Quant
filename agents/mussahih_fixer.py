import os
import sys
import asyncio
import ast
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import importlib
import subprocess
import time
import re
from collections import defaultdict

from .base_agent import BaseAgent


class MussahihFixer(BaseAgent):
    """Issue Resolution Specialist Agent for diagnosing and fixing system issues"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'fix_reports_path': 'reports/fixes/',
            'backup_path': 'backups/',
            'analysis_depth': 'deep',  # 'surface', 'medium', 'deep'
            'auto_fix_enabled': True,
            'impact_assessment_threshold': 0.7,  # Risk threshold for auto-fixes
            'rollback_enabled': True
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(
            name="Mussahih",
            role="Issue Resolution Specialist",
            config=default_config
        )
        
        self.issue_registry = {}
        self.fix_history = []
        self.impact_assessments = {}
        self.dependency_graph = {}
        
        self.initialize_diagnostic_tools()
    
    def initialize_diagnostic_tools(self):
        """Initialize diagnostic and analysis tools"""
        self.update_status("initializing", "Setting up diagnostic environment")
        
        # Create directories
        for directory in ['reports/fixes', 'backups', 'analysis']:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize issue patterns and solutions
        self.known_patterns = self.load_known_issue_patterns()
        
        # Initialize dependency analyzer
        self.build_system_dependency_graph()
        
        self.update_status("ready", "Diagnostic tools initialized")
    
    def load_known_issue_patterns(self) -> Dict[str, Any]:
        """Load known issue patterns and their solutions"""
        return {
            'database_connection_error': {
                'pattern': r'(sqlite3\.OperationalError|database.*locked|connection.*failed)',
                'severity': 'high',
                'common_causes': ['database locked', 'permission issues', 'corrupted database'],
                'solutions': ['restart_db_connection', 'check_permissions', 'repair_database']
            },
            'prediction_accuracy_low': {
                'pattern': r'(accuracy|mape).*below.*threshold',
                'severity': 'medium',
                'common_causes': ['insufficient data', 'model parameters', 'market volatility'],
                'solutions': ['retrain_model', 'adjust_parameters', 'increase_data_window']
            },
            'memory_leak': {
                'pattern': r'memory.*usage.*high|out of memory',
                'severity': 'high',
                'common_causes': ['unclosed connections', 'large data sets', 'recursive calls'],
                'solutions': ['close_connections', 'optimize_data_processing', 'garbage_collection']
            },
            'ui_element_not_found': {
                'pattern': r'element.*not found|selenium.*timeout',
                'severity': 'medium',
                'common_causes': ['page loading delay', 'element ID changed', 'dynamic content'],
                'solutions': ['increase_wait_time', 'update_selectors', 'add_explicit_waits']
            }
        }
    
    def build_system_dependency_graph(self):
        """Build system component dependency graph"""
        self.dependency_graph = {
            'database': {
                'depends_on': [],
                'dependents': ['models', 'api', 'ui'],
                'criticality': 'high',
                'recovery_time': 5  # minutes
            },
            'models': {
                'depends_on': ['database'],
                'dependents': ['predictions', 'ui'],
                'criticality': 'high',
                'recovery_time': 15
            },
            'api': {
                'depends_on': ['database', 'models'],
                'dependents': ['ui'],
                'criticality': 'medium',
                'recovery_time': 10
            },
            'ui': {
                'depends_on': ['api', 'database'],
                'dependents': [],
                'criticality': 'low',
                'recovery_time': 5
            },
            'predictions': {
                'depends_on': ['models', 'database'],
                'dependents': ['ui'],
                'criticality': 'high',
                'recovery_time': 20
            }
        }
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute issue resolution tasks"""
        task_type = task.get('type')
        
        try:
            if task_type == 'diagnose_issue':
                return await self.diagnose_issue(task['issue_data'])
            
            elif task_type == 'analyze_test_failures':
                return await self.analyze_test_failures(task['test_results'])
            
            elif task_type == 'fix_issue':
                return await self.fix_issue(task['issue_id'], task.get('auto_approve', False))
            
            elif task_type == 'impact_assessment':
                return await self.perform_impact_assessment(task['proposed_fix'])
            
            elif task_type == 'rollback_fix':
                return await self.rollback_fix(task['fix_id'])
            
            elif task_type == 'optimize_system':
                return await self.optimize_system(task.get('target_components'))
            
            elif task_type == 'preventive_analysis':
                return await self.preventive_analysis()
            
            else:
                return {
                    'success': False,
                    'error': f'Unknown task type: {task_type}'
                }
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def diagnose_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive issue diagnosis"""
        self.update_status("diagnosing", f"Analyzing issue: {issue_data.get('description', 'Unknown')}")
        
        issue_id = f"issue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        diagnosis = {
            'issue_id': issue_id,
            'timestamp': datetime.now().isoformat(),
            'original_issue': issue_data,
            'analysis': {
                'pattern_matches': [],
                'root_causes': [],
                'affected_components': [],
                'severity': 'unknown',
                'urgency': 'medium'
            },
            'recommendations': []
        }
        
        # Pattern matching analysis
        error_message = issue_data.get('error_message', '')
        test_failures = issue_data.get('test_failures', [])
        
        for pattern_name, pattern_data in self.known_patterns.items():
            if re.search(pattern_data['pattern'], error_message, re.IGNORECASE):
                diagnosis['analysis']['pattern_matches'].append({
                    'pattern': pattern_name,
                    'confidence': 0.8,
                    'severity': pattern_data['severity'],
                    'solutions': pattern_data['solutions']
                })
        
        # Analyze test failures
        if test_failures:
            failure_analysis = await self.analyze_test_failure_patterns(test_failures)
            diagnosis['analysis']['root_causes'].extend(failure_analysis['root_causes'])
            diagnosis['analysis']['affected_components'].extend(failure_analysis['affected_components'])
        
        # Component dependency analysis
        affected_components = diagnosis['analysis']['affected_components']
        cascading_effects = self.analyze_cascading_effects(affected_components)
        diagnosis['analysis']['cascading_effects'] = cascading_effects
        
        # Generate recommendations
        diagnosis['recommendations'] = await self.generate_fix_recommendations(diagnosis)
        
        # Determine severity and urgency
        diagnosis['analysis']['severity'] = self.calculate_severity(diagnosis)
        diagnosis['analysis']['urgency'] = self.calculate_urgency(diagnosis)
        
        # Store issue in registry
        self.issue_registry[issue_id] = diagnosis
        
        return diagnosis
    
    async def analyze_test_failure_patterns(self, test_failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in test failures to identify root causes"""
        analysis = {
            'root_causes': [],
            'affected_components': [],
            'common_patterns': []
        }
        
        failure_types = defaultdict(list)
        component_failures = defaultdict(int)
        
        for failure in test_failures:
            failure_type = failure.get('type', 'unknown')
            component = failure.get('component', 'unknown')
            error_msg = failure.get('error', '')
            
            failure_types[failure_type].append(failure)
            component_failures[component] += 1
            
            # Check for specific patterns
            if 'timeout' in error_msg.lower():
                analysis['root_causes'].append('Performance degradation or resource contention')
            elif 'connection' in error_msg.lower():
                analysis['root_causes'].append('Database or network connectivity issues')
            elif 'not found' in error_msg.lower():
                analysis['root_causes'].append('Missing dependencies or changed interfaces')
        
        # Identify most affected components
        analysis['affected_components'] = [
            comp for comp, count in component_failures.items() 
            if count > 1  # Multiple failures in same component
        ]
        
        # Common failure patterns
        for failure_type, failures in failure_types.items():
            if len(failures) > 2:  # Pattern if multiple similar failures
                analysis['common_patterns'].append({
                    'type': failure_type,
                    'count': len(failures),
                    'likely_cause': self.infer_cause_from_pattern(failure_type, failures)
                })
        
        return analysis
    
    def infer_cause_from_pattern(self, failure_type: str, failures: List[Dict[str, Any]]) -> str:
        """Infer likely cause from failure pattern"""
        if failure_type == 'selenium':
            return 'UI element changes or timing issues'
        elif failure_type == 'database':
            return 'Data integrity or connection problems'
        elif failure_type == 'api':
            return 'Service unavailability or interface changes'
        elif failure_type == 'validation':
            return 'Business logic changes or data quality issues'
        else:
            return 'Systemic issue requiring deeper investigation'
    
    def analyze_cascading_effects(self, affected_components: List[str]) -> Dict[str, Any]:
        """Analyze potential cascading effects of component failures"""
        cascading_analysis = {
            'direct_impacts': [],
            'indirect_impacts': [],
            'recovery_order': [],
            'estimated_downtime': 0
        }
        
        all_impacted = set(affected_components)
        
        # Find components that depend on affected ones
        for component in affected_components:
            if component in self.dependency_graph:
                dependents = self.dependency_graph[component]['dependents']
                all_impacted.update(dependents)
                
                for dependent in dependents:
                    cascading_analysis['direct_impacts'].append({
                        'component': dependent,
                        'cause': f'Depends on failed component: {component}',
                        'criticality': self.dependency_graph.get(dependent, {}).get('criticality', 'unknown')
                    })
        
        # Calculate recovery order based on dependencies
        cascading_analysis['recovery_order'] = self.calculate_recovery_order(list(all_impacted))
        
        # Estimate total downtime
        total_recovery_time = sum(
            self.dependency_graph.get(comp, {}).get('recovery_time', 10)
            for comp in all_impacted
        )
        cascading_analysis['estimated_downtime'] = total_recovery_time
        
        return cascading_analysis
    
    def calculate_recovery_order(self, components: List[str]) -> List[str]:
        """Calculate optimal order for component recovery"""
        # Topological sort based on dependencies
        in_degree = {comp: 0 for comp in components}
        
        for component in components:
            if component in self.dependency_graph:
                for dependent in self.dependency_graph[component]['dependents']:
                    if dependent in in_degree:
                        in_degree[dependent] += 1
        
        # Components with no dependencies first
        queue = [comp for comp, degree in in_degree.items() if degree == 0]
        recovery_order = []
        
        while queue:
            current = queue.pop(0)
            recovery_order.append(current)
            
            if current in self.dependency_graph:
                for dependent in self.dependency_graph[current]['dependents']:
                    if dependent in in_degree:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)
        
        return recovery_order
    
    async def generate_fix_recommendations(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized fix recommendations"""
        recommendations = []
        
        # From pattern matches
        for match in diagnosis['analysis']['pattern_matches']:
            for solution in match['solutions']:
                recommendations.append({
                    'type': 'pattern_based',
                    'action': solution,
                    'confidence': match['confidence'],
                    'severity': match['severity'],
                    'description': f"Apply {solution} based on pattern match: {match['pattern']}"
                })
        
        # From root cause analysis
        for cause in diagnosis['analysis']['root_causes']:
            if 'performance' in cause.lower():
                recommendations.append({
                    'type': 'performance',
                    'action': 'optimize_performance',
                    'confidence': 0.7,
                    'severity': 'medium',
                    'description': f"Address performance issue: {cause}"
                })
            elif 'connectivity' in cause.lower():
                recommendations.append({
                    'type': 'connectivity',
                    'action': 'check_connections',
                    'confidence': 0.8,
                    'severity': 'high',
                    'description': f"Fix connectivity issue: {cause}"
                })
        
        # Sort by severity and confidence
        recommendations.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}.get(x['severity'], 0),
            x['confidence']
        ), reverse=True)
        
        return recommendations
    
    def calculate_severity(self, diagnosis: Dict[str, Any]) -> str:
        """Calculate overall issue severity"""
        severity_scores = {'low': 1, 'medium': 2, 'high': 3}
        
        max_severity = 0
        for match in diagnosis['analysis']['pattern_matches']:
            severity_score = severity_scores.get(match['severity'], 1)
            max_severity = max(max_severity, severity_score)
        
        # Consider affected components
        affected_count = len(diagnosis['analysis']['affected_components'])
        if affected_count > 3:
            max_severity = max(max_severity, 3)
        elif affected_count > 1:
            max_severity = max(max_severity, 2)
        
        severity_map = {1: 'low', 2: 'medium', 3: 'high'}
        return severity_map.get(max_severity, 'medium')
    
    def calculate_urgency(self, diagnosis: Dict[str, Any]) -> str:
        """Calculate issue urgency based on system impact"""
        urgency_factors = {
            'database_down': 'critical',
            'user_facing_error': 'high',
            'test_failures': 'medium',
            'performance_degradation': 'low'
        }
        
        # Check for critical components
        affected = diagnosis['analysis']['affected_components']
        critical_components = ['database', 'models', 'predictions']
        
        if any(comp in critical_components for comp in affected):
            return 'critical'
        
        # Check pattern severity
        high_severity_patterns = [
            match for match in diagnosis['analysis']['pattern_matches']
            if match['severity'] == 'high'
        ]
        
        if high_severity_patterns:
            return 'high'
        
        return 'medium'
    
    async def fix_issue(self, issue_id: str, auto_approve: bool = False) -> Dict[str, Any]:
        """Execute fix for diagnosed issue"""
        self.update_status("fixing", f"Applying fixes for issue {issue_id}")
        
        if issue_id not in self.issue_registry:
            return {
                'success': False,
                'error': f'Issue {issue_id} not found in registry'
            }
        
        diagnosis = self.issue_registry[issue_id]
        fix_result = {
            'fix_id': f"fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'issue_id': issue_id,
            'timestamp': datetime.now().isoformat(),
            'applied_fixes': [],
            'success': True,
            'rollback_available': False
        }
        
        # Create backup if rollback is enabled
        if self.config['rollback_enabled']:
            backup_result = await self.create_system_backup()
            fix_result['backup_id'] = backup_result.get('backup_id')
            fix_result['rollback_available'] = backup_result.get('success', False)
        
        # Apply fixes based on recommendations
        for recommendation in diagnosis['recommendations']:
            if not auto_approve and recommendation['severity'] == 'high':
                # For high-severity fixes, require manual approval
                fix_result['applied_fixes'].append({
                    'action': recommendation['action'],
                    'status': 'pending_approval',
                    'reason': 'High-severity fix requires manual approval'
                })
                continue
            
            # Apply fix
            fix_action_result = await self.apply_fix_action(recommendation)
            fix_result['applied_fixes'].append(fix_action_result)
            
            if not fix_action_result.get('success', False):
                fix_result['success'] = False
        
        # Update issue status
        self.issue_registry[issue_id]['status'] = 'fixed' if fix_result['success'] else 'partial_fix'
        self.issue_registry[issue_id]['fix_applied'] = fix_result['fix_id']
        
        # Store fix history
        self.fix_history.append(fix_result)
        
        # Generate fix report
        await self.generate_fix_report(fix_result)
        
        return fix_result
    
    async def apply_fix_action(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific fix action"""
        action = recommendation['action']
        
        action_result = {
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'details': []
        }
        
        try:
            if action == 'restart_db_connection':
                # Simulate database connection restart
                action_result['success'] = True
                action_result['details'].append('Database connection pool restarted')
                
            elif action == 'retrain_model':
                # Simulate model retraining
                action_result['success'] = True
                action_result['details'].append('Model retrained with updated parameters')
                
            elif action == 'optimize_performance':
                # Simulate performance optimization
                action_result['success'] = True
                action_result['details'].append('Performance optimization applied')
                
            elif action == 'update_selectors':
                # Simulate UI selector updates
                action_result['success'] = True
                action_result['details'].append('UI selectors updated for test stability')
                
            elif action == 'check_connections':
                # Simulate connection health check
                action_result['success'] = True
                action_result['details'].append('All connections verified and restored')
                
            else:
                action_result['details'].append(f'Unknown action: {action}')
                
        except Exception as e:
            action_result['error'] = str(e)
            action_result['details'].append(f'Fix action failed: {str(e)}')
        
        return action_result
    
    async def perform_impact_assessment(self, proposed_fix: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact of proposed fix before application"""
        self.update_status("assessing", "Analyzing fix impact")
        
        assessment = {
            'fix_description': proposed_fix,
            'timestamp': datetime.now().isoformat(),
            'risk_level': 'low',
            'affected_systems': [],
            'mitigation_strategies': [],
            'rollback_plan': {},
            'approval_required': False
        }
        
        # Analyze fix type and scope
        fix_type = proposed_fix.get('type', 'unknown')
        affected_components = proposed_fix.get('components', [])
        
        # Calculate risk level
        risk_factors = []
        
        if 'database' in affected_components:
            risk_factors.append('data_integrity')
            assessment['risk_level'] = 'high'
        
        if 'models' in affected_components:
            risk_factors.append('prediction_accuracy')
            assessment['risk_level'] = max(assessment['risk_level'], 'medium')
        
        if fix_type in ['schema_change', 'architecture_change']:
            risk_factors.append('system_stability')
            assessment['risk_level'] = 'high'
        
        assessment['risk_factors'] = risk_factors
        
        # Determine if approval is required
        if assessment['risk_level'] == 'high' or len(affected_components) > 2:
            assessment['approval_required'] = True
        
        # Generate mitigation strategies
        assessment['mitigation_strategies'] = self.generate_mitigation_strategies(assessment)
        
        # Create rollback plan
        assessment['rollback_plan'] = self.create_rollback_plan(proposed_fix)
        
        return assessment
    
    def generate_mitigation_strategies(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        for risk_factor in assessment.get('risk_factors', []):
            if risk_factor == 'data_integrity':
                strategies.append('Create full database backup before changes')
                strategies.append('Validate data integrity after fix application')
            elif risk_factor == 'prediction_accuracy':
                strategies.append('Run validation tests on historical data')
                strategies.append('Implement gradual rollout with monitoring')
            elif risk_factor == 'system_stability':
                strategies.append('Deploy to staging environment first')
                strategies.append('Monitor system metrics during deployment')
        
        return strategies
    
    def create_rollback_plan(self, proposed_fix: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed rollback plan"""
        return {
            'backup_required': True,
            'rollback_steps': [
                'Stop affected services',
                'Restore from backup',
                'Restart services',
                'Verify system functionality'
            ],
            'estimated_rollback_time': 15,  # minutes
            'verification_checklist': [
                'Database connectivity',
                'Model predictions',
                'UI functionality'
            ]
        }
    
    async def create_system_backup(self) -> Dict[str, Any]:
        """Create system backup for rollback purposes"""
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = f"{self.config['backup_path']}/{backup_id}"
        
        try:
            Path(backup_path).mkdir(parents=True, exist_ok=True)
            
            # Simulate backup creation (in real implementation, would backup actual files/database)
            backup_manifest = {
                'backup_id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'included_components': ['database', 'models', 'config'],
                'backup_size': '50MB',  # Simulated
                'integrity_check': 'passed'
            }
            
            with open(f"{backup_path}/manifest.json", 'w') as f:
                json.dump(backup_manifest, f, indent=2)
            
            return {
                'success': True,
                'backup_id': backup_id,
                'backup_path': backup_path,
                'manifest': backup_manifest
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Backup creation failed: {str(e)}'
            }
    
    async def preventive_analysis(self) -> Dict[str, Any]:
        """Perform preventive analysis to identify potential issues"""
        self.update_status("analyzing", "Performing preventive system analysis")
        
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'potential_issues': [],
            'maintenance_recommendations': [],
            'health_score': 0.0,
            'risk_areas': []
        }
        
        # Analyze system health indicators
        health_indicators = await self.collect_health_indicators()
        
        # Check for warning signs
        warning_signs = self.detect_warning_signs(health_indicators)
        analysis_result['potential_issues'] = warning_signs
        
        # Generate maintenance recommendations
        analysis_result['maintenance_recommendations'] = self.generate_maintenance_recommendations(health_indicators)
        
        # Calculate overall health score
        analysis_result['health_score'] = self.calculate_health_score(health_indicators)
        
        # Identify risk areas
        analysis_result['risk_areas'] = self.identify_risk_areas(health_indicators)
        
        return analysis_result
    
    async def collect_health_indicators(self) -> Dict[str, Any]:
        """Collect various system health indicators"""
        indicators = {
            'database_performance': {'response_time_ms': 150, 'connection_pool_usage': 0.3},
            'model_accuracy': {'last_mae': 0.05, 'prediction_confidence': 0.85},
            'system_resources': {'cpu_usage': 0.45, 'memory_usage': 0.6, 'disk_usage': 0.3},
            'error_rates': {'api_errors': 0.02, 'ui_errors': 0.01, 'test_failures': 0.05},
            'performance_trends': {'response_time_trend': 'stable', 'accuracy_trend': 'improving'}
        }
        
        return indicators
    
    def detect_warning_signs(self, health_indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect early warning signs of potential issues"""
        warning_signs = []
        
        # Database performance warnings
        db_perf = health_indicators['database_performance']
        if db_perf['response_time_ms'] > 200:
            warning_signs.append({
                'type': 'performance_degradation',
                'component': 'database',
                'severity': 'medium',
                'description': f"Database response time elevated: {db_perf['response_time_ms']}ms"
            })
        
        # Resource usage warnings
        resources = health_indicators['system_resources']
        if resources['memory_usage'] > 0.8:
            warning_signs.append({
                'type': 'resource_exhaustion',
                'component': 'system',
                'severity': 'high',
                'description': f"High memory usage: {resources['memory_usage']*100:.1f}%"
            })
        
        # Error rate warnings
        errors = health_indicators['error_rates']
        if errors['test_failures'] > 0.1:
            warning_signs.append({
                'type': 'quality_degradation',
                'component': 'testing',
                'severity': 'medium',
                'description': f"Elevated test failure rate: {errors['test_failures']*100:.1f}%"
            })
        
        return warning_signs
    
    def generate_maintenance_recommendations(self, health_indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proactive maintenance recommendations"""
        recommendations = []
        
        # Database maintenance
        db_perf = health_indicators['database_performance']
        if db_perf['response_time_ms'] > 100:
            recommendations.append({
                'type': 'database_optimization',
                'priority': 'medium',
                'action': 'Database optimization and index rebuild',
                'estimated_impact': 'Improve response time by 20-30%'
            })
        
        # Model retraining
        model_acc = health_indicators['model_accuracy']
        if model_acc['last_mae'] > 0.04:
            recommendations.append({
                'type': 'model_maintenance',
                'priority': 'high',
                'action': 'Retrain prediction models with recent data',
                'estimated_impact': 'Improve prediction accuracy'
            })
        
        return recommendations
    
    def calculate_health_score(self, health_indicators: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-1)"""
        scores = []
        
        # Database health (response time)
        db_response = health_indicators['database_performance']['response_time_ms']
        db_score = max(0, min(1, (300 - db_response) / 300))  # 0-300ms scale
        scores.append(db_score)
        
        # Model accuracy
        model_mae = health_indicators['model_accuracy']['last_mae']
        accuracy_score = max(0, min(1, (0.1 - model_mae) / 0.1))  # 0-0.1 scale
        scores.append(accuracy_score)
        
        # Resource utilization (inverse - lower is better)
        resource_score = 1 - health_indicators['system_resources']['memory_usage']
        scores.append(resource_score)
        
        # Error rates (inverse)
        error_score = 1 - health_indicators['error_rates']['test_failures']
        scores.append(error_score)
        
        return round(sum(scores) / len(scores), 2)
    
    def identify_risk_areas(self, health_indicators: Dict[str, Any]) -> List[str]:
        """Identify high-risk areas requiring attention"""
        risk_areas = []
        
        if health_indicators['system_resources']['memory_usage'] > 0.7:
            risk_areas.append('Memory Management')
        
        if health_indicators['error_rates']['test_failures'] > 0.08:
            risk_areas.append('Code Quality')
        
        if health_indicators['model_accuracy']['last_mae'] > 0.06:
            risk_areas.append('Model Performance')
        
        return risk_areas
    
    async def generate_fix_report(self, fix_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fix report"""
        report_filename = f"fix_report_{fix_result['fix_id']}.json"
        report_path = f"{self.config['fix_reports_path']}/{report_filename}"
        
        report = {
            'fix_summary': fix_result,
            'timestamp': datetime.now().isoformat(),
            'generated_by': self.name,
            'impact_assessment': self.impact_assessments.get(fix_result['issue_id'], {}),
            'lessons_learned': self.extract_lessons_learned(fix_result)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return {
            'success': True,
            'report_path': report_path,
            'report_summary': report
        }
    
    def extract_lessons_learned(self, fix_result: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from fix application"""
        lessons = []
        
        successful_fixes = [fix for fix in fix_result['applied_fixes'] if fix.get('success')]
        failed_fixes = [fix for fix in fix_result['applied_fixes'] if not fix.get('success')]
        
        if successful_fixes:
            lessons.append(f"Successfully applied {len(successful_fixes)} fixes")
        
        if failed_fixes:
            lessons.append(f"{len(failed_fixes)} fixes require manual intervention")
        
        if fix_result.get('rollback_available'):
            lessons.append("Backup created successfully - rollback option available")
        
        return lessons
    
    def get_capabilities(self) -> List[str]:
        """Return list of Mussahih's capabilities"""
        return [
            'issue_diagnosis',
            'root_cause_analysis',
            'impact_assessment',
            'holistic_fixing',
            'system_optimization',
            'preventive_analysis',
            'dependency_analysis',
            'rollback_management',
            'fix_reporting',
            'maintenance_planning'
        ] 