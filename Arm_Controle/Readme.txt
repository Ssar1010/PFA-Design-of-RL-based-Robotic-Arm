# 2-DOF Robotic Arm Control on Raspberry Pi via DQN

## Project Overview

Hi! This project demonstrates how I deployed my **Deep Q-Network (DQN) trained policy** to control a **2-degree-of-freedom robotic arm** on a **Raspberry Pi 3**.

The robotic arm has:
- **Servo 1 (base):** rotates 0–180°
- **Servo 2 (arm/joint):** rotates 0–90°
- **Ultrasonic sensor** mounted at the end-effector to measure distance to targets

The goal is to reach **real-world targets** using the sensor readings, applying the policy trained in simulation.

---

## Hardware & Setup

- **Raspberry Pi 3**  
- **Two servo motors** connected to GPIO pins  
- **Ultrasonic sensor** (HC-SR04 or similar)  
- **RPi.GPIO** library for PWM control  
- **Torch** to load the trained `best.pt` policy  

---

## How It Works

1. **Policy Loading**  
   - Rebuild the same neural network structure used in training.  
   - Load the trained weights from `best.pt`.  

2. **Environment Representation**  
   - Input to the network: `[theta1, theta2, measured_distance, target_distance]`  
   - Output: four discrete actions to adjust the servo angles.  

3. **Action Execution**  
   - Base servo: ±0.15 rad per step  
   - Arm servo: ±0.15 rad per step  
   - Actions are clamped to physical servo limits  

4. **Target Reaching**  
   - Distance measured by ultrasonic sensor is used as the reward signal  
   - Stops moving when the target is reached or max steps are exceeded  

5. **Servo Control**  
   - Angles are converted to PWM duty cycles  
   - Servos move with a small delay for smooth motion  

---

## Usage

1. Connect the servos and ultrasonic sensor to the Raspberry Pi GPIO pins.  
2. Place `best.pt` in the same directory as the script.  
3. Run the Python script:

```bash
python run_rpi_arm_dqn.py
The arm will attempt **5 randomly generated targets**, moving in real-time.

---

## Results

- The arm **successfully reaches multiple targets** using the ultrasonic sensor.  
- The policy trained in simulation **transfers to real hardware**.  
- Demonstrates **real-world generalization** of my RL policy.

---

## Notes & Tips

- Adjust `time.sleep()` for faster or smoother servo movement.  
- Ensure servo angles **do not exceed physical limits**.  
- The network runs on **CPU**, fully compatible with Raspberry Pi 3.  
- Ultrasonic sensor measurements can be noisy; consider **averaging readings** for smoother control.

---

## References & Libraries

- **PyTorch** — Neural network inference: [https://pytorch.org](https://pytorch.org)  
- **RPi.GPIO** — Servo PWM control: [https://pypi.org/project/RPi.GPIO/](https://pypi.org/project/RPi.GPIO/)  
- **NumPy** — Computations and kinematics: [https://numpy.org](https://numpy.org)
