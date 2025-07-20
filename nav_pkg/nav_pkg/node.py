import sys
import pygame
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout, QTextEdit,
    QCheckBox, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, Circle

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HazardPublisherNode(Node):
    def __init__(self, gui_callback=None):
        super().__init__('hazard_rover_controller')
        
        # Publishers
        self.wheel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.emergency_pub = self.create_publisher(String, 'emergency_alert', 10)
        
        # Subscribers
        self.hazard_sub = self.create_subscription(
            Float32,
            '/uwb/hazard_distance',
            self.hazard_callback,
            10
        )
        
        logger.info("Hazard Rover Controller Node initialized")
        self.active = True
        self.gui_callback = gui_callback
        self.hazard_threshold = 5.0  # meters
        self.current_distance = 999.0  # Initialize with safe distance
        self.emergency_triggered = False
        self.safe_mode_enabled = False  # New: Safe mode instead of complete lockout

    def hazard_callback(self, msg):
        """Callback for hazard distance updates"""
        if not self.active:
            return
            
        self.current_distance = msg.data
        
        # Check if distance breached threshold
        if self.current_distance <= self.hazard_threshold:
            if not self.emergency_triggered:
                self.trigger_emergency()
        else:
            if self.emergency_triggered:
                self.clear_emergency()
        
        # Update GUI if callback provided
        if self.gui_callback:
            self.gui_callback(self.current_distance, self.emergency_triggered)

    def trigger_emergency(self):
        """Trigger emergency state - enable safe mode instead of complete lockout"""
        self.emergency_triggered = True
        self.safe_mode_enabled = True
        
        # Send emergency stop command initially
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.wheel_pub.publish(stop_msg)
        
        # Send emergency alert
        alert_msg = String()
        alert_msg.data = f"EMERGENCY: Hazard distance breached! Distance: {self.current_distance:.2f}m - SAFE MODE ENABLED"
        self.emergency_pub.publish(alert_msg)
        
        self.get_logger().error(f"EMERGENCY TRIGGERED: Distance {self.current_distance:.2f}m <= {self.hazard_threshold}m - Safe mode enabled")

    def clear_emergency(self):
        """Clear emergency state"""
        self.emergency_triggered = False
        self.safe_mode_enabled = False
        
        # Send all clear alert
        alert_msg = String()
        alert_msg.data = f"ALL CLEAR: Safe distance restored. Distance: {self.current_distance:.2f}m - Normal operation resumed"
        self.emergency_pub.publish(alert_msg)
        
        self.get_logger().info(f"EMERGENCY CLEARED: Distance {self.current_distance:.2f}m > {self.hazard_threshold}m")

    def wheel_drive(self, x: float, z: float, speed: float, override_safe_mode=False):
        """Send wheel drive commands with smart safety restrictions"""
        if not self.active:
            return
            
        # In safe mode, allow only backward movement and turning
        if self.safe_mode_enabled and not override_safe_mode:
            # Allow backward movement (negative x) and turning (any z)
            if x > 0:  # Block forward movement
                self.get_logger().warning("Forward movement blocked in safe mode! Use backward or turn to escape.")
                x = 0.0  # Set forward movement to zero
            
            # Reduce speed in safe mode for safety
            speed = min(speed, 0.3)  # Max 30% speed in safe mode
            
            # Log the restriction
            if x == 0.0 and speed > 0:
                self.get_logger().info("Safe mode: Forward blocked, backward/turn allowed")
            
        try:
            msg = Twist()
            msg.linear.x = x * speed
            msg.angular.z = z * speed
            self.wheel_pub.publish(msg)
            
            mode_str = "SAFE MODE" if self.safe_mode_enabled else "NORMAL"
            self.get_logger().info(f"WHEEL [{mode_str}]: x: {x:.2f} | z: {z:.2f} | speed: {speed:.2f}")
            
        except Exception as e:
            logger.error(f"Publishing error: {e}")

    def set_threshold(self, threshold: float):
        """Update hazard threshold"""
        self.hazard_threshold = threshold
        self.get_logger().info(f"Hazard threshold updated to: {threshold:.2f}m")

    def disable_safe_mode(self):
        """Temporarily disable safe mode (emergency override)"""
        if self.emergency_triggered:
            self.safe_mode_enabled = False
            self.get_logger().warning("SAFE MODE DISABLED - Manual override activated!")
            
            # Send alert
            alert_msg = String()
            alert_msg.data = "WARNING: Safe mode manually disabled - Full control enabled"
            self.emergency_pub.publish(alert_msg)

    def enable_safe_mode(self):
        """Re-enable safe mode"""
        if self.emergency_triggered:
            self.safe_mode_enabled = True
            self.get_logger().info("Safe mode re-enabled")

    def shutdown(self):
        self.active = False
        self.destroy_node()

class HazardControlGUI(QMainWindow):
    # Signal for thread-safe GUI updates
    hazard_update_signal = pyqtSignal(float, bool)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mars Rover Hazard Control System - Enhanced Safety")
        self.setGeometry(100, 100, 1400, 800)  # Increased window size
        self.shutting_down = False

        # Initialize pygame for gamepad support
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.init_gamepad()

        # Control variables
        self.speed = 0.0
        self.x_dir = 0.0  # Forward/backward direction
        self.z_dir = 0.0  # Left/right steering
        self.button_states = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False
        }

        # Hazard monitoring variables
        self.hazard_distance = 999.0
        self.emergency_active = False
        self.hazard_threshold = 5.0
        self.safe_mode_override = False  # Manual override for safe mode

        # ROS2 setup
        self.ros_node = None
        self.ros_connected = False
        self.setup_ros_node()

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Use a horizontal layout for the main window with three columns
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel (controls) - 60% width
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setSpacing(15)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Middle panel (logs) - 20% width
        self.middle_panel = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_panel)
        self.middle_layout.setSpacing(15)
        self.middle_layout.setContentsMargins(0, 0, 0, 0)
        
        # Right panel (plot) - 20% width
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setSpacing(15)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Initialize components
        self.init_hazard_display()
        self.init_safety_controls()
        self.init_controls()
        self.init_logs()
        self.init_plot()
        
        # Add panels to main layout with stretch factors
        self.main_layout.addWidget(self.left_panel, 60)  # 60% width
        self.main_layout.addWidget(self.middle_panel, 20)  # 20% width
        self.main_layout.addWidget(self.right_panel, 20)  # 20% width

        # Connect signal for thread-safe updates
        self.hazard_update_signal.connect(self.update_hazard_display)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_output)
        self.timer.start(50)  # 20Hz update
        
        self.gamepad_timer = QTimer()
        self.gamepad_timer.timeout.connect(self.update_gamepad)
        self.gamepad_timer.start(100)  # 10Hz update

        # Track last published values
        self.last_x = 0.0
        self.last_z = 0.0
        self.last_speed = 0.0
        
        # Deadzone for change detection
        self.deadzone = 0.01  # 1% change threshold

    def init_hazard_display(self):
        """Initialize hazard monitoring display"""
        hazard_group = QGroupBox("Hazard Monitoring System")
        hazard_layout = QVBoxLayout()
        
        # Status indicators row
        status_row = QHBoxLayout()
        
        # Hazard status light
        self.hazard_light = QLabel("●")
        self.hazard_light.setStyleSheet("""
            QLabel {
                font-size: 48px;
                color: #27ae60;
                font-weight: bold;
                padding: 10px;
                border-radius: 25px;
                background-color: #ecf0f1;
                min-width: 80px;
                text-align: center;
            }
        """)
        self.hazard_light.setAlignment(Qt.AlignCenter)
        
        # Hazard info panel
        info_layout = QVBoxLayout()
        self.distance_label = QLabel("Distance: ---.-- m")
        self.distance_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        
        self.status_label = QLabel("Status: SAFE")
        self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; font-weight: bold;")
        
        self.mode_label = QLabel("Mode: NORMAL OPERATION")
        self.mode_label.setStyleSheet("font-size: 14px; color: #2c3e50; font-weight: bold;")
        
        self.threshold_label = QLabel(f"Threshold: {self.hazard_threshold:.1f} m")
        self.threshold_label.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        
        info_layout.addWidget(self.distance_label)
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.mode_label)
        info_layout.addWidget(self.threshold_label)
        
        status_row.addWidget(self.hazard_light)
        status_row.addLayout(info_layout)
        status_row.addStretch()
        
        # Threshold control
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Emergency Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 100)  # 1.0 to 10.0 meters (scaled by 10)
        self.threshold_slider.setValue(int(self.hazard_threshold * 10))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        self.threshold_value_label = QLabel(f"{self.hazard_threshold:.1f} m")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        
        hazard_layout.addLayout(status_row)
        hazard_layout.addLayout(threshold_layout)
        hazard_group.setLayout(hazard_layout)
        
        hazard_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #34495e;
                border-radius: 10px;
                margin-top: 1.5ex;
                padding: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #2c3e50;
                font-size: 14px;
            }
        """)
        
        self.left_layout.addWidget(hazard_group)

    def init_safety_controls(self):
        """Initialize enhanced safety control panel"""
        safety_group = QGroupBox("Safety Override Controls")
        safety_layout = QVBoxLayout()
        
        # Safe mode explanation
        explanation = QLabel("""
        🔒 SAFE MODE: When hazard is detected, forward movement is blocked but backward movement and turning are allowed.
        ⚠️  OVERRIDE: Temporarily disable safe mode restrictions (use with extreme caution!).
        """)
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 10px;")
        safety_layout.addWidget(explanation)
        
        # Override controls
        override_layout = QHBoxLayout()
        
        self.safe_mode_checkbox = QCheckBox("Enable Safe Mode Override")
        self.safe_mode_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e74c3c;
                font-weight: bold;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #e74c3c;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #e74c3c;
                background-color: #e74c3c;
            }
        """)
        self.safe_mode_checkbox.stateChanged.connect(self.toggle_safe_mode_override)
        
        self.override_status = QLabel("Safe Mode: ENABLED")
        self.override_status.setStyleSheet("color: #27ae60; font-weight: bold; padding-left: 20px;")
        
        override_layout.addWidget(self.safe_mode_checkbox)
        override_layout.addWidget(self.override_status)
        override_layout.addStretch()
        
        safety_layout.addLayout(override_layout)
        
        safety_group.setLayout(safety_layout)
        safety_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #e74c3c;
                border-radius: 10px;
                margin-top: 1.5ex;
                padding: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #e74c3c;
                font-size: 14px;
            }
        """)
        
        self.left_layout.addWidget(safety_group)

    def toggle_safe_mode_override(self, state):
        """Toggle safe mode override"""
        self.safe_mode_override = state == Qt.Checked
        
        if self.safe_mode_override:
            self.override_status.setText("Safe Mode: OVERRIDDEN ⚠️")
            self.override_status.setStyleSheet("color: #e74c3c; font-weight: bold; padding-left: 20px;")
            if self.ros_node:
                self.ros_node.disable_safe_mode()
            self.add_log("WARNING: Safe mode override ENABLED - Full movement allowed")
        else:
            self.override_status.setText("Safe Mode: ENABLED")
            self.override_status.setStyleSheet("color: #27ae60; font-weight: bold; padding-left: 20px;")
            if self.ros_node:
                self.ros_node.enable_safe_mode()
            self.add_log("Safe mode override DISABLED - Safety restrictions active")

    def init_gamepad(self):
        """Initialize gamepad if available"""
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            try:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                logger.info(f"Gamepad connected: {self.joystick.get_name()}")
                logger.info(f"Gamepad axes: {self.joystick.get_numaxes()}")
                logger.info(f"Gamepad buttons: {self.joystick.get_numbuttons()}")
            except Exception as e:
                logger.error(f"Gamepad init error: {e}")
                self.joystick = None
        else:
            logger.warning("No gamepad detected")
            self.joystick = None

    def setup_ros_node(self):
        """Setup ROS2 node in a separate thread"""
        def ros_thread_function():
            try:
                rclpy.init()
                self.ros_node = HazardPublisherNode(gui_callback=self.hazard_callback_gui)
                logger.info("ROS2 hazard node initialized")
                self.ros_connected = True
                
                # Spin until shutdown
                while rclpy.ok() and not self.shutting_down:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.1)
                
                if self.ros_node:
                    self.ros_node.shutdown()
                rclpy.shutdown()
                self.ros_connected = False
            except Exception as e:
                logger.error(f"ROS node error: {e}")
                self.ros_connected = False

        self.ros_thread = threading.Thread(target=ros_thread_function, daemon=True)
        self.ros_thread.start()

    def hazard_callback_gui(self, distance, emergency_active):
        """Thread-safe callback from ROS node"""
        self.hazard_update_signal.emit(distance, emergency_active)

    def update_hazard_display(self, distance, emergency_active):
        """Update hazard display (called from signal)"""
        self.hazard_distance = distance
        self.emergency_active = emergency_active
        
        # Update distance display
        self.distance_label.setText(f"Distance: {distance:.2f} m")
        
        if emergency_active:
            # Orange/red light for hazard detected
            self.hazard_light.setStyleSheet("""
                QLabel {
                    font-size: 48px;
                    color: #f39c12;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 25px;
                    background-color: #fef9e7;
                    min-width: 80px;
                    text-align: center;
                    border: 3px solid #f39c12;
                }
            """)
            self.status_label.setText("Status: HAZARD DETECTED")
            self.status_label.setStyleSheet("font-size: 16px; color: #f39c12; font-weight: bold;")
            
            # Update mode display based on override status
            if self.safe_mode_override:
                self.mode_label.setText("Mode: OVERRIDE ACTIVE ⚠️")
                self.mode_label.setStyleSheet("font-size: 14px; color: #e74c3c; font-weight: bold;")
            else:
                self.mode_label.setText("Mode: SAFE MODE (Backward/Turn Only)")
                self.mode_label.setStyleSheet("font-size: 14px; color: #f39c12; font-weight: bold;")
            
            # Flash effect for hazard
            if hasattr(self, 'flash_timer'):
                self.flash_timer.stop()
            self.flash_timer = QTimer()
            self.flash_timer.timeout.connect(self.flash_hazard_light)
            self.flash_timer.start(750)  # Flash every 750ms
            
        else:
            # Green light for safe
            self.hazard_light.setStyleSheet("""
                QLabel {
                    font-size: 48px;
                    color: #27ae60;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 25px;
                    background-color: #d5f4e6;
                    min-width: 80px;
                    text-align: center;
                    border: 3px solid #27ae60;
                }
            """)
            self.status_label.setText("Status: SAFE")
            self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; font-weight: bold;")
            
            self.mode_label.setText("Mode: NORMAL OPERATION")
            self.mode_label.setStyleSheet("font-size: 14px; color: #2c3e50; font-weight: bold;")
            
            # Stop flashing
            if hasattr(self, 'flash_timer'):
                self.flash_timer.stop()
                
            # Reset override checkbox when safe
            if not self.safe_mode_override:
                self.safe_mode_checkbox.setChecked(False)

    def flash_hazard_light(self):
        """Flash the hazard light"""
        current_style = self.hazard_light.styleSheet()
        if "#f39c12" in current_style:
            # Change to darker orange
            self.hazard_light.setStyleSheet("""
                QLabel {
                    font-size: 48px;
                    color: #e67e22;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 25px;
                    background-color: #fcf3cf;
                    min-width: 80px;
                    text-align: center;
                    border: 3px solid #e67e22;
                }
            """)
        else:
            # Change back to bright orange
            self.hazard_light.setStyleSheet("""
                QLabel {
                    font-size: 48px;
                    color: #f39c12;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 25px;
                    background-color: #fef9e7;
                    min-width: 80px;
                    text-align: center;
                    border: 3px solid #f39c12;
                }
            """)

    def update_threshold(self, value):
        """Update hazard threshold"""
        self.hazard_threshold = value / 10.0  # Convert back from scaled value
        self.threshold_value_label.setText(f"{self.hazard_threshold:.1f} m")
        self.threshold_label.setText(f"Threshold: {self.hazard_threshold:.1f} m")
        
        # Update ROS node threshold
        if self.ros_node:
            self.ros_node.set_threshold(self.hazard_threshold)

    def init_controls(self):
        """Initialize control interface"""
        # Status indicators
        status_layout = QHBoxLayout()
        self.ros_status = QLabel("🔴 ROS Disconnected")
        self.ros_status.setStyleSheet("font-weight: bold; padding: 5px; color: red;")
        self.gamepad_status = QLabel("🔴 No Gamepad")
        self.gamepad_status.setStyleSheet("font-weight: bold; padding: 5px; color: red;")
        status_layout.addWidget(self.ros_status)
        status_layout.addWidget(self.gamepad_status)
        self.left_layout.addLayout(status_layout)

        # Speed control
        speed_group = QGroupBox("Speed Control")
        speed_layout = QVBoxLayout()
        
        self.speed_label = QLabel("0%")
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_label.setStyleSheet("font-weight: bold; color: #2b2d42;")
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(0)
        self.speed_slider.setMinimumHeight(35)
        self.speed_slider.valueChanged.connect(self.update_speed)
        
        # Style the slider
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #cccccc;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #2b2d42;
                width: 30px;
                height: 30px;
                margin: -10px 0;
                border-radius: 15px;
            }
        """)
        
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_group.setLayout(speed_layout)
        self.left_layout.addWidget(speed_group)

        # Direction buttons
        btn_group = QGroupBox("Direction Control")
        grid = QGridLayout()
        grid.setSpacing(15)
        grid.setContentsMargins(20, 20, 20, 20)
        
        # Create buttons
        self.btn_forward = QPushButton("↑ FORWARD")
        self.btn_backward = QPushButton("↓ BACKWARD") 
        self.btn_left = QPushButton("← LEFT")
        self.btn_right = QPushButton("→ RIGHT")
        self.btn_stop = QPushButton("EMERGENCY STOP")
        
        # Set button styles
        for btn in [self.btn_forward, self.btn_backward, self.btn_left, self.btn_right]:
            btn.setMinimumHeight(60)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2b2d42; 
                    color: white; 
                    border-radius: 10px; 
                    padding: 15px; 
                    font-weight: bold;
                    font-size: 16px;
                }
                QPushButton:pressed {
                    background-color: #4a4d6d;
                }
                QPushButton:disabled {
                    background-color: #7f8c8d;
                    color: #bdc3c7;
                }
            """)
        
        # Special styling for forward button when in safe mode
        self.btn_forward.setProperty("safe_mode_blocked", False)
        
        self.btn_stop.setMinimumHeight(60)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; 
                color: white; 
                border-radius: 10px; 
                padding: 15px; 
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """)
        
        # Connect signals
        self.btn_forward.pressed.connect(lambda: self.set_direction_button('forward', True))
        self.btn_forward.released.connect(lambda: self.set_direction_button('forward', False))
        self.btn_backward.pressed.connect(lambda: self.set_direction_button('backward', True))
        self.btn_backward.released.connect(lambda: self.set_direction_button('backward', False))
        self.btn_left.pressed.connect(lambda: self.set_direction_button('left', True))
        self.btn_left.released.connect(lambda: self.set_direction_button('left', False))
        self.btn_right.pressed.connect(lambda: self.set_direction_button('right', True))
        self.btn_right.released.connect(lambda: self.set_direction_button('right', False))
        self.btn_stop.clicked.connect(self.emergency_stop)
        
        # Layout
        grid.addWidget(self.btn_forward, 0, 1)
        grid.addWidget(self.btn_left, 1, 0)
        grid.addWidget(self.btn_stop, 1, 1)
        grid.addWidget(self.btn_right, 1, 2)
        grid.addWidget(self.btn_backward, 2, 1)
        
        btn_group.setLayout(grid)
        btn_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 8px;
                margin-top: 1.5ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        self.left_layout.addWidget(btn_group)

    def init_plot(self):
        """Initialize the direction visualization plot"""
        plot_group = QGroupBox("Movement Direction")
        plot_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("Movement Direction", fontsize=10, fontweight='bold')
        self.ax.set_xlabel("Left/Right", fontsize=10)
        self.ax.set_ylabel("Forward/Backward", fontsize=10)
        
        # Add center point and direction arrow
        self.center_circle = Circle((0, 0), 0.05, color='#3498db', zorder=5)
        self.ax.add_patch(self.center_circle)
        
        # Add hazard zone indicator
        self.hazard_zone = Circle((0, 1.2), 0.3, color='#e74c3c', alpha=0.3, zorder=1)
        self.ax.add_patch(self.hazard_zone)
        self.ax.text(0, 1.2, 'HAZARD', ha='center', va='center', fontsize=8, fontweight='bold', color='#e74c3c')
        
        # Initial arrow (will be updated)
        self.direction_arrow = None
        
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        self.right_layout.addWidget(plot_group)
        
        # Add stretch to push everything up
        self.right_layout.addStretch()

    def init_logs(self):
        """Initialize log display"""
        log_group = QGroupBox("System Logs")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #f4f6f7;
                color: #2c3e50;
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                min-height: 300px;
            }
        """)
        
        # Clear logs button
        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.clear_logs_btn.clicked.connect(self.log_display.clear)
        
        log_layout.addWidget(self.log_display)
        log_layout.addWidget(self.clear_logs_btn)
        
        log_group.setLayout(log_layout)
        log_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 1.5ex;
                padding: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        self.middle_layout.addWidget(log_group)
        
        # Add stretch to push everything up
        self.middle_layout.addStretch()

    def add_log(self, message: str):
        """Append message to log display"""
        self.log_display.append(f"[LOG] {message}")
        logger.info(message)

    def update_speed(self, value):
        """Update speed from slider"""
        self.speed = value / 100.0  # normalize 0-1
        self.speed_label.setText(f"{value}%")
        self.add_log(f"Speed set to {value}%")

    def update_output(self):
        """Send wheel commands at regular interval"""
        # Update status indicators
        ros_connected = self.ros_node is not None and self.ros_node.active
        gamepad_connected = self.joystick is not None
        
        self.ros_status.setText("🟢 ROS Connected" if ros_connected else "🔴 ROS Disconnected")
        self.ros_status.setStyleSheet(f"font-weight: bold; padding: 5px; color: {'green' if ros_connected else 'red'};")
        
        self.gamepad_status.setText("🟢 Gamepad Connected" if gamepad_connected else "🔴 No Gamepad")
        self.gamepad_status.setStyleSheet(f"font-weight: bold; padding: 5px; color: {'green' if gamepad_connected else 'red'};")
        
        if not self.ros_node:
            return
        
        x = 0.0
        z = 0.0
        if self.button_states['forward']:
            x += 1.0
        if self.button_states['backward']:
            x -= 1.0
        if self.button_states['left']:
            z += 1.0
        if self.button_states['right']:
            z -= 1.0
        
        # Only send if significant change
        if abs(x - self.last_x) > self.deadzone or abs(z - self.last_z) > self.deadzone or abs(self.speed - self.last_speed) > self.deadzone:
            self.ros_node.wheel_drive(x, z, self.speed, override_safe_mode=self.safe_mode_override)
            self.last_x = x
            self.last_z = z
            self.last_speed = self.speed

    def set_direction_button(self, direction, pressed):
        """Handle direction button press/release"""
        self.button_states[direction] = pressed
        self.add_log(f"Button {direction} {'pressed' if pressed else 'released'}")

    def emergency_stop(self):
        """Send stop command"""
        self.ros_node.wheel_drive(0.0, 0.0, 0.0)
        self.add_log("EMERGENCY STOP issued")

    def update_gamepad(self):
        """Read gamepad inputs if connected"""
        # Try to reconnect if gamepad not detected
        if not self.joystick:
            try:
                pygame.joystick.quit()
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                    logger.info(f"Gamepad reconnected: {self.joystick.get_name()}")
            except Exception as e:
                logger.error(f"Gamepad reconnect error: {e}")
                self.joystick = None
            return
        
        try:
            pygame.event.pump()
            
            # Get axis values safely
            num_axes = self.joystick.get_numaxes()
            left_x = 0.0
            left_y = 0.0
            right_x = 0.0  # Right stick X for speed control

            if num_axes > 3:
                right_x = self.joystick.get_axis(3)
            if num_axes > 2:
                left_x = self.joystick.get_axis(2)
            
            if num_axes > 0:
                left_x = self.joystick.get_axis(0)
            if num_axes > 1:
                left_y = self.joystick.get_axis(1)
            
            # Update speed from right stick
            current_slider_val = self.speed_slider.value()
            change = int(right_x * 5)  # tweak step size as needed
            new_speed_val = max(0, min(100, current_slider_val + change))
            self.speed_slider.setValue(new_speed_val)
            self.speed = new_speed_val / 100.0
            
            # Calculate direction from left stick
            deadzone = 0.2
            self.x_dir = -left_y if abs(left_y) > deadzone else 0.0
            self.z_dir = left_x if abs(left_x) > deadzone else 0.0
            
            # Normalize direction vector
            magnitude = np.sqrt(self.x_dir**2 + self.z_dir**2)
            if magnitude > 1.0:
                self.x_dir /= magnitude
                self.z_dir /= magnitude
                
            # START button to stop
            if self.joystick.get_button(9):  # START button
                self.emergency_stop()
                
            # Update plot
            self.update_plot()
            
        except Exception as e:
            logger.error(f"Gamepad error: {e}")
            self.joystick = None

    def closeEvent(self, event):
        """Handle window close"""
        self.shutting_down = True
        if self.ros_node:
            self.ros_node.shutdown()
        pygame.quit()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = HazardControlGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()