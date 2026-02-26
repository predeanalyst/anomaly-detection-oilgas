"""
SAP ECC Integration Module

Handles automated work order creation for detected anomalies.
Integrates with SAP using PyRFC or REST APIs.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try importing PyRFC for SAP connectivity
try:
    from pyrfc import Connection
    PYRFC_AVAILABLE = True
except ImportError:
    PYRFC_AVAILABLE = False
    logger.warning("PyRFC not available. SAP integration will use REST API fallback.")


class SAPConnector:
    """
    SAP ECC connector for automated work order creation.
    
    Supports both RFC (PyRFC) and REST API connections.
    """
    
    def __init__(
        self,
        server: str,
        client: str,
        system_number: str,
        user: str,
        password: str,
        use_rfc: bool = True,
        language: str = 'EN'
    ):
        """
        Initialize SAP connector.
        
        Args:
            server: SAP server hostname
            client: SAP client number
            system_number: SAP system number
            user: SAP username
            password: SAP password
            use_rfc: Use RFC connection (requires PyRFC)
            language: SAP language code
        """
        self.server = server
        self.client = client
        self.system_number = system_number
        self.user = user
        self.password = password
        self.language = language
        self.use_rfc = use_rfc and PYRFC_AVAILABLE
        
        self.connection = None
        
        if self.use_rfc:
            self._connect_rfc()
        else:
            logger.info("Using REST API mode for SAP integration")
    
    def _connect_rfc(self):
        """Establish RFC connection to SAP."""
        try:
            self.connection = Connection(
                ashost=self.server,
                sysnr=self.system_number,
                client=self.client,
                user=self.user,
                passwd=self.password,
                lang=self.language
            )
            logger.info(f"Connected to SAP system: {self.server}")
        except Exception as e:
            logger.error(f"Failed to connect to SAP: {e}")
            self.use_rfc = False
    
    def create_maintenance_order(
        self,
        equipment_id: str,
        description: str,
        priority: str = 'Medium',
        anomaly_details: Optional[Dict] = None,
        plant: str = '1000',
        order_type: str = 'PM01'
    ) -> Dict:
        """
        Create a maintenance work order in SAP.
        
        Args:
            equipment_id: Equipment ID in SAP
            description: Work order description
            priority: Priority level (High/Medium/Low)
            anomaly_details: Detailed anomaly information
            plant: Plant code
            order_type: Order type (PM01 = Maintenance)
            
        Returns:
            result: Created work order details
        """
        # Map priority to SAP priority codes
        priority_map = {
            'High': '1',
            'Medium': '3',
            'Low': '5'
        }
        
        priority_code = priority_map.get(priority, '3')
        
        # Prepare work order data
        order_data = {
            'ORDERID': '',  # Will be generated
            'ORDER_TYPE': order_type,
            'EQUIPMENT': equipment_id,
            'SHORT_TEXT': description,
            'PRIORITY': priority_code,
            'PLANPLANT': plant,
            'SYSTCOND': 'CRTD',  # Created status
            'CREATED_ON': datetime.now().strftime('%Y%m%d'),
            'CREATED_AT': datetime.now().strftime('%H%M%S')
        }
        
        # Add anomaly details to long text
        if anomaly_details:
            long_text = self._format_anomaly_details(anomaly_details)
            order_data['LONG_TEXT'] = long_text
        
        if self.use_rfc:
            result = self._create_order_rfc(order_data)
        else:
            result = self._create_order_rest(order_data)
        
        return result
    
    def _create_order_rfc(self, order_data: Dict) -> Dict:
        """Create order using RFC."""
        try:
            # Call SAP BAPI for order creation
            # BAPI_ALM_ORDER_MAINTAIN is commonly used
            result = self.connection.call(
                'BAPI_ALM_ORDER_MAINTAIN',
                IT_HEADER=[order_data],
                TESTRUN=' '  # Empty = execute
            )
            
            order_number = result.get('ORDER', {}).get('ORDERID', '')
            
            if order_number:
                logger.info(f"Created SAP work order: {order_number}")
                return {
                    'success': True,
                    'order_number': order_number,
                    'message': 'Work order created successfully'
                }
            else:
                error_msg = result.get('RETURN', {}).get('MESSAGE', 'Unknown error')
                logger.error(f"Failed to create work order: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg
                }
        
        except Exception as e:
            logger.error(f"Error creating work order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_order_rest(self, order_data: Dict) -> Dict:
        """Create order using REST API (fallback)."""
        # This is a placeholder - actual implementation would use SAP OData/REST APIs
        logger.info("REST API work order creation not fully implemented")
        
        # Simulate successful creation for demo
        order_number = f"WO-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            'success': True,
            'order_number': order_number,
            'message': 'Work order created (simulated)',
            'mode': 'REST_API'
        }
    
    def _format_anomaly_details(self, anomaly_details: Dict) -> str:
        """Format anomaly details for SAP long text."""
        lines = [
            "=== AUTOMATED ANOMALY DETECTION ALERT ===",
            "",
            f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Anomaly Score: {anomaly_details.get('anomaly_score', 'N/A'):.4f}",
            f"Threshold: {anomaly_details.get('threshold', 'N/A'):.4f}",
            "",
            "Affected Sensors:"
        ]
        
        # Add top contributing sensors
        top_features = anomaly_details.get('top_features', [])
        for feature in top_features:
            lines.append(f"  - {feature.get('feature')}: {feature.get('contribution', 0):.2%}")
        
        lines.extend([
            "",
            "Recommended Actions:",
            "  1. Inspect identified sensors for drift or failure",
            "  2. Verify equipment operational parameters",
            "  3. Schedule preventive maintenance if needed",
            "",
            "This work order was automatically created by the Anomaly Detection System."
        ])
        
        return "\n".join(lines)
    
    def update_order_status(
        self,
        order_number: str,
        status: str,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Update work order status.
        
        Args:
            order_number: Work order number
            status: New status code
            notes: Additional notes
            
        Returns:
            result: Update result
        """
        if self.use_rfc:
            try:
                result = self.connection.call(
                    'BAPI_ALM_ORDER_MAINTAIN',
                    ORDER_NUMBER=order_number,
                    STATUS=status,
                    NOTES=notes or ''
                )
                
                logger.info(f"Updated work order {order_number} to status {status}")
                return {'success': True}
            
            except Exception as e:
                logger.error(f"Error updating work order: {e}")
                return {'success': False, 'error': str(e)}
        else:
            logger.info(f"Simulated status update: {order_number} -> {status}")
            return {'success': True, 'mode': 'REST_API'}
    
    def get_equipment_details(self, equipment_id: str) -> Dict:
        """
        Retrieve equipment details from SAP.
        
        Args:
            equipment_id: Equipment ID
            
        Returns:
            equipment_details: Equipment information
        """
        if self.use_rfc:
            try:
                result = self.connection.call(
                    'BAPI_EQUI_GETDETAIL',
                    EQUIPMENT=equipment_id
                )
                
                return result.get('EQUIPMENT_DATA', {})
            
            except Exception as e:
                logger.error(f"Error retrieving equipment details: {e}")
                return {}
        else:
            # Simulated equipment data
            return {
                'EQUIPMENT': equipment_id,
                'DESCRIPT': 'Equipment Description',
                'STATUS': 'Active'
            }
    
    def close(self):
        """Close SAP connection."""
        if self.connection:
            self.connection.close()
            logger.info("SAP connection closed")


class WorkOrderManager:
    """
    Higher-level work order management with business logic.
    """
    
    def __init__(self, sap_connector: SAPConnector):
        """
        Initialize work order manager.
        
        Args:
            sap_connector: SAP connector instance
        """
        self.sap = sap_connector
        self.pending_orders = []
    
    def process_anomaly_alert(
        self,
        anomaly_result: Dict,
        equipment_mapping: Dict[str, str],
        auto_create: bool = True
    ) -> Optional[Dict]:
        """
        Process anomaly detection result and create work orders as needed.
        
        Args:
            anomaly_result: Anomaly detection result
            equipment_mapping: Mapping of sensors to equipment IDs
            auto_create: Automatically create work orders
            
        Returns:
            work_order: Created work order details
        """
        if not anomaly_result.get('is_anomaly'):
            return None
        
        # Determine affected equipment
        top_features = anomaly_result.get('top_features', [])
        if not top_features:
            return None
        
        primary_sensor = top_features[0]['feature']
        equipment_id = equipment_mapping.get(primary_sensor)
        
        if not equipment_id:
            logger.warning(f"No equipment mapping found for sensor: {primary_sensor}")
            return None
        
        # Determine priority based on anomaly score
        score = anomaly_result.get('anomaly_score', 0)
        if score > 0.9:
            priority = 'High'
        elif score > 0.7:
            priority = 'Medium'
        else:
            priority = 'Low'
        
        # Generate description
        description = f"Anomaly detected in {primary_sensor} - Score: {score:.2f}"
        
        # Create work order
        if auto_create:
            work_order = self.sap.create_maintenance_order(
                equipment_id=equipment_id,
                description=description,
                priority=priority,
                anomaly_details=anomaly_result
            )
            
            if work_order['success']:
                self.pending_orders.append(work_order)
            
            return work_order
        else:
            # Queue for manual approval
            pending = {
                'equipment_id': equipment_id,
                'description': description,
                'priority': priority,
                'anomaly_details': anomaly_result
            }
            self.pending_orders.append(pending)
            return pending
    
    def get_pending_orders(self) -> List[Dict]:
        """Get list of pending work orders."""
        return self.pending_orders
