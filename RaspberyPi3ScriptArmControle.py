# Raspberry Pi 3 - 2-DOF Arm + Ultrasonic Sensor Inference
# Requirements: torch, RPi.GPIO, numpy

import torch
import torch.nn as nn
import numpy as np
import time
import RPi.GPIO as GPIO

# ---------------------------
# GPIO setup
# ---------------------------
GPIO.setmode(GPIO.BCM)

# Servo pins
BASE_PIN = 17
ARM_PIN = 27

# Ultrasonic sensor pins
TRIG_PIN = 23
ECHO_PIN = 24

GPIO.setup(BASE_PIN, GPIO.OUT)
GPIO.setup(ARM_PIN, GPIO.OUT)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# PWM for servos
base_pwm = GPIO.PWM(BASE_PIN, 50)
arm_pwm = GPIO.PWM(ARM_PIN, 50)
base_pwm.start(7.5)
arm_pwm.start(7.5)

# ---------------------------
# Servo helper functions
# ---------------------------
def angle_to_duty(angle, min_angle=0, max_angle=180, min_duty=2.5, max_duty=12.5):
    """Convert angle in degrees to duty cycle for servo."""
    duty = min_duty + (angle - min_angle) * (max_duty - min_duty) / (max_angle - min_angle)
    return duty

def move_servos(theta1, theta2):
    """Move the two servos to desired angles."""
    duty1 = angle_to_duty(np.degrees(theta1), 0, 180)
    duty2 = angle_to_duty(np.degrees(theta2), 0, 90)
    base_pwm.ChangeDutyCycle(duty1)
    arm_pwm.ChangeDutyCycle(duty2)
    time.sleep(0.3)

# ---------------------------
# Ultrasonic sensor helper
# ---------------------------
def get_distance():
    """Return distance measured by ultrasonic sensor in meters."""
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.05)
    
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    pulse_start, pulse_end = time.time(), time.time()
    timeout = time.time() + 0.04  # 40ms timeout

    while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout:
        pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 343 / 2  # Speed of sound = 343 m/s
    return distance

# ---------------------------
# DQN Policy Network
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load trained policy
model = PolicyNet()
model.load_state_dict(torch.load("best.pt", map_location=torch.device('cpu')))
model.eval()

# ---------------------------
# Forward kinematics
# ---------------------------
theta1 = np.radians(90)
theta2 = np.radians(45)
l1, l2 = 1.0, 1.0

def forward_kinematics(t1, t2):
    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    return np.array([x, y])

# ---------------------------
# Main control loop
# ---------------------------
try:
    for episode in range(5):
        print(f"Episode {episode+1} - Using ultrasonic sensor for distance")

        for step in range(60):
            distance = get_distance()  # distance in meters
            target_distance = 0.2  # desired distance to "target" object

            # Build observation: [theta1, theta2, measured_distance, target_distance]
            obs = torch.tensor([theta1, theta2, distance, target_distance], dtype=torch.float32)

            with torch.no_grad():
                q_values = model(obs)
                action = torch.argmax(q_values).item()

            # Map action to servo angles
            if action == 0: theta1 += 0.15
            elif action == 1: theta1 -= 0.15
            elif action == 2: theta2 += 0.15
            elif action == 3: theta2 -= 0.15

            # Clamp angles
            theta1 = np.clip(theta1, 0, np.pi)
            theta2 = np.clip(theta2, 0, np.pi/2)

            move_servos(theta1, theta2)

            # Stop if within threshold distance
            if abs(distance - target_distance) < 0.02:
                print(f"Target reached! Distance: {distance:.2f} m")
                break

        time.sleep(1)

finally:
    base_pwm.stop()
    arm_pwm.stop()
    GPIO.cleanup()
    print("Program finished and GPIO cleaned up.")
