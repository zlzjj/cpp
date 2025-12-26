# Simple Car Task

A simple navigation task for a differential-drive car in MuJoCo MPC.

## Description

This task demonstrates a basic car navigation controller where a simple car with two rear wheels and a front caster wheel must navigate to a randomly moving goal position. When the car reaches within 0.2 units of the goal, the goal automatically relocates to a new random position within the arena.

## Model

The car model (`car_model.xml`) features:
- **Chassis**: A custom mesh representing the car body with explicit inertial properties
- **Wheels**: Two rear wheels controlled independently via tendons
- **Front caster**: A sphere providing forward support
- **Actuators**: Two motors for forward/backward motion and turning
  - `forward`: Controls average speed of both wheels (gear=12)
  - `turn`: Controls differential speed for steering (gear=6)
- **Physics**: 
  - Newton solver with 50 iterations for numerical stability
  - 2ms timestep
  - Enhanced friction (1.5 for wheels, 1.0 for ground)
  - Joint damping (0.05) and armature (0.005) for stability
  - **Maximum speed**: ~1.5-2.0 m/s

## Task Definition

The task is defined in `simple_car.cc` with the following residuals:

### Residuals (4 total)
1. **Position X** (residual 0): Distance from goal x-coordinate
2. **Position Y** (residual 1): Distance from goal y-coordinate  
3. **Control Forward** (residual 2): Forward motor control effort
4. **Control Turn** (residual 3): Turn motor control effort

### Cost Function

The cost function penalizes:
- Distance to goal position (x, y)
- Control effort for both actuators

### Goal Behavior

The goal is a green semi-transparent sphere (mocap body) that:
- Starts at position (1.0, 1.0, 0.01)
- Moves randomly within range [-2.0, 2.0] for both x and y when car gets within 0.2 units
- Always stays at ground level (z = 0.01)

This creates a continuous navigation task where the car must constantly adapt to new goal locations.

## Running the Task

```bash
./build/bin/mjpc --task SimpleCar
```

### Viewing Speed Information

The car's speed is displayed as a white text label **"Speed: X.XX m/s (Y.Y km/h)"** floating above the car in the 3D scene.

The speed is calculated as the magnitude of the car's linear velocity vector, with automatic conversion to km/h (multiply by 3.6).

At maximum speed (~1.5-2.0 m/s), the car travels at approximately **5.4-7.2 km/h**.

**Alternative**: You can also press **F5** to view the sensor panel (bottom right) which shows detailed sensor readings including `car_velocity` and `car_angular_velocity`.

## Files

- `car_model.xml`: MuJoCo model definition for the car
- `task.xml`: Task configuration including cost weights and parameters
- `simple_car.h`: Task header file
- `simple_car.cc`: Task implementation with residual function
