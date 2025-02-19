import numpy as np
import control as ct

# Define the nonlinear dynamics for the vehicle steering system
def steering_update(t, x, u, params):
    """
    Nonlinear dynamics of the vehicle steering system.
    
    Parameters:
    t : float
        Current time (not used in the dynamics, but included for compatibility).
    x : array_like
        Current state vector: [x1, x2, x3].
    u : array_like
        Input vector: [velocity, steering angle].
    params : dict
        Dictionary of parameters (a, b, maxsteer).
    
    Returns:
    dx : array_like
        State derivatives: [dx1, dx2, dx3].
    """
    a = params.get('a', 1.5)         # Offset to vehicle reference point [m]
    b = params.get('b', 3.0)         # Vehicle wheelbase [m]
    maxsteer = params.get('maxsteer', 0.5)  # Max steering angle [rad]
    
    # Saturate the steering input
    delta = np.clip(u[1], -maxsteer, maxsteer)
    
    # System dynamics
    alpha = np.arctan2(a * np.tan(delta), b)
    dx1 = u[0] * np.cos(x[2] + alpha)  # x velocity
    dx2 = u[0] * np.sin(x[2] + alpha)  # y velocity
    dx3 = (u[0] / b) * np.tan(delta)   # Angular velocity
    return [dx1, dx2, dx3]

# Create a nonlinear input/output system
steering = ct.nlsys(
    steering_update,                    # Update function for system dynamics
    inputs=['v', 'delta'],              # Inputs: velocity and steering angle
    outputs=None,                       # Outputs are the same as the states
    states=['x', 'y', 'theta'],         # States: x, y, and theta (angle)
    name='steering',
    params={'a': 1.5, 'b': 3.0, 'maxsteer': 0.5}  # Default parameters
)

# Generate the linearization at a given velocity
def linearize_lateral(v0=10, normalize=False, output_full_state=False):
    # Compute the linearization at the given velocity
    linsys = ct.linearize(steering, 0, [v0, 0])

    # Extract out the lateral dynamics
    latsys = ct.model_reduction(
        linsys, elim_states=[0], keep_inputs=[1], keep_outputs=[1],
        method='truncate', warn_unstable=False)

    # Normalize coordinates if desired
    if normalize:
        b = steering.params['b']
        latsys = ct.similarity_transform(
            latsys, np.array([[1/b, 0], [0, 1]]), timescale=v0/b)

    C = np.eye(2) if output_full_state else np.array([[1, 0]])

    # Normalized system with (normalized) lateral offset as output
    return ct.ss(
        latsys.A, latsys.B, C, 0, inputs='delta',
        outputs=['y', 'theta'] if output_full_state else 'y')
