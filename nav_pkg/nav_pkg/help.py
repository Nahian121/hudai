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
    QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, Circle

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HazardControlGUI(QMainWindow):
    hazard_update_signal = pyqtSignal(float, bool)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mars Rover Hazard Control System")
        self.setGeometry(100, 100, 1200, 800)  # Adjusted window size
        
        # Gamepad variables
        self.joystick = None
        self.gamepad_connected = False
        self.last_gamepad_check = 0
        
        # Initialize pygame for gamepad
        pygame.init()
        pygame.joystick.init()
        self.init_gamepad()

        # Control variables
        self.speed = 0.0
        self.x_dir = 0.0
        self.z_dir = 0.0
        self.button_states = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False
        }

        # Hazard monitoring
        self.hazard_distance = 999.0
        self.emergency_active = False
        self.hazard_threshold = 5.0
        self.safe_mode_override = False

        # ROS setup
        self.ros_node = None
        self.ros_connected = False
        self.setup_ros_node()

        # Main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel (60%)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setSpacing(10)
        
        # Middle panel (20%)
        self.middle_panel = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_panel)
        self.middle_layout.setSpacing(10)
        
        # Right panel (20% - larger for square plot)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setSpacing(10)
        
        # Initialize components
        self.init_hazard_display()
        self.init_safety_controls()
        self.init_controls()
        self.init_logs()
        self.init_plot()
        
        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel, 60)
        self.main_layout.addWidget(self.middle_panel, 20)
        self.main_layout.addWidget(self.right_panel, 20)

        # Connect signals
        self.hazard_update_signal.connect(self.update_hazard_display)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_output)
        self.timer.start(50)  # 20Hz
        
        self.gamepad_timer = QTimer()
        self.gamepad_timer.timeout.connect(self.update_gamepad)
        self.gamepad_timer.start(500)  # Check gamepad every 500ms

        # Track last values
        self.last_x = 0.0
        self.last_z = 0.0
        self.last_speed = 0.0
        self.deadzone = 0.01

    def init_gamepad(self):
        """Initialize gamepad if available"""
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.gamepad_connected = True
                logger.info(f"Gamepad connected: {self.joystick.get_name()}")
            else:
                self.gamepad_connected = False
                self.joystick = None
        except Exception as e:
            logger.error(f"Gamepad init error: {e}")
            self.gamepad_connected = False
            self.joystick = None

    def setup_ros_node(self):
        """Setup ROS2 node in a separate thread"""
        def ros_thread_function():
            try:
                rclpy.init()
                self.ros_node = HazardPublisherNode(gui_callback=self.hazard_callback_gui)
                self.ros_connected = True
                logger.info("ROS2 node initialized")
                
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

    def init_hazard_display(self):
        """Initialize hazard monitoring display"""
        hazard_group = QGroupBox("Hazard Monitoring System")
        hazard_layout = QVBoxLayout()
        
        # Status indicators
        status_row = QHBoxLayout()
        
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
        
        info_layout = QVBoxLayout()
        self.distance_label = QLabel("Distance: ---.-- m")
        self.distance_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        self.status_label = QLabel("Status: SAFE")
        self.status_label.setStyleSheet("font-size: 16px; color: #27ae60; font-weight: bold;")
        
        self.mode_label = QLabel("Mode: NORMAL OPERATION")
        self.mode_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.threshold_label = QLabel(f"Threshold: {self.hazard_threshold:.1f} m")
        
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
        self.threshold_slider.setRange(10, 100)
        self.threshold_slider.setValue(int(self.hazard_threshold * 10))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        self.threshold_value_label = QLabel(f"{self.hazard_threshold:.1f} m")
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        
        hazard_layout.addLayout(status_row)
        hazard_layout.addLayout(threshold_layout)
        hazard_group.setLayout(hazard_layout)
        
        self.left_layout.addWidget(hazard_group)

    def init_plot(self):
        """Initialize the direction visualization plot"""
        plot_group = QGroupBox("Movement Direction")
        plot_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(6, 6))  # Larger square figure
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Set square aspect ratio
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("Movement Direction", fontsize=10, fontweight='bold')
        
        # Add center point
        self.center_circle = Circle((0, 0), 0.05, color='#3498db', zorder=5)
        self.ax.add_patch(self.center_circle)
        
        # Add hazard zone indicator
        self.hazard_zone = Circle((0, 1.2), 0.3, color='#e74c3c', alpha=0.3, zorder=1)
        self.ax.add_patch(self.hazard_zone)
        self.ax.text(0, 1.2, 'HAZARD', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='#e74c3c')
        
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        
        # Set size policy to make it expand
        plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_layout.addWidget(plot_group)

    def update_gamepad(self):
        """Check and update gamepad status"""
        try:
            # Only check gamepad status every 5 seconds to prevent flickering
            current_time = pygame.time.get_ticks()
            if current_time - self.last_gamepad_check > 5000:  # 5 seconds
                self.last_gamepad_check = current_time
                was_connected = self.gamepad_connected
                self.init_gamepad()  # This updates self.gamepad_connected
                
                # Only log if status changed
                if was_connected != self.gamepad_connected:
                    if self.gamepad_connected:
                        logger.info("Gamepad connected")
                    else:
                        logger.info("Gamepad disconnected")
            
            if not self.gamepad_connected:
                return
                
            pygame.event.pump()
            
            # Get axis values
            num_axes = self.joystick.get_numaxes()
            if num_axes > 0:
                left_x = self.joystick.get_axis(0)
                left_y = self.joystick.get_axis(1) if num_axes > 1 else 0.0
                right_x = self.joystick.get_axis(3) if num_axes > 3 else 0.0
                
                # Update speed from right stick
                current_slider_val = self.speed_slider.value()
                change = int(right_x * 5)
                new_speed_val = max(0, min(100, current_slider_val + change))
                self.speed_slider.setValue(new_speed_val)
                self.speed = new_speed_val / 100.0
                
                # Calculate direction from left stick
                deadzone = 0.2
                self.x_dir = -left_y if abs(left_y) > deadzone else 0.0
                self.z_dir = left_x if abs(left_x) > deadzone else 0.0
                
                # Normalize
                magnitude = np.sqrt(self.x_dir**2 + self.z_dir**2)
                if magnitude > 1.0:
                    self.x_dir /= magnitude
                    self.z_dir /= magnitude
                    
                # START button to stop
                if self.joystick.get_button(9):
                    self.emergency_stop()
                    
                self.update_plot()
                
        except Exception as e:
            logger.error(f"Gamepad error: {e}")
            self.gamepad_connected = False
            self.joystick = None

    def update_output(self):
        """Update status and send commands"""
        # Update connection status
        self.ros_status.setText("🟢 ROS Connected" if self.ros_connected else "🔴 ROS Disconnected")
        self.ros_status.setStyleSheet(f"color: {'green' if self.ros_connected else 'red'}; font-weight: bold;")
        
        self.gamepad_status.setText("🟢 Gamepad Connected" if self.gamepad_connected else "🔴 No Gamepad")
        self.gamepad_status.setStyleSheet(f"color: {'green' if self.gamepad_connected else 'red'}; font-weight: bold;")
        
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
        
        if (abs(x - self.last_x) > self.deadzone or 
            abs(z - self.last_z) > self.deadzone or 
            abs(self.speed - self.last_speed) > self.deadzone):
            
            self.ros_node.wheel_drive(x, z, self.speed, override_safe_mode=self.safe_mode_override)
            self.last_x = x
            self.last_z = z
            self.last_speed = self.speed

    # ... (rest of the methods remain the same as in the previous implementation)

class HazardPublisherNode(Node):
    # ... (keep the same implementation as before)

def main():
    app = QApplication(sys.argv)
    window = HazardControlGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()