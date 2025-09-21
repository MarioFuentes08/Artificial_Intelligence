import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Sensor ranges
sensor_range = 40


def calculate_wall_distance(x, y, rad_sensor_angle, max_range):
    """
    Calculate the distance to the nearest wall in the direction of the sensor.
    
    This function computes the intersection point between a sensor ray (originating
    from the robot position) and the environment walls, returning the closest
    valid intersection distance.
    
    Parameters
    ----------
    x : float
        The x-coordinate of the robot's current position (0-100)
    y : float  
        The y-coordinate of the robot's current position (0-100)
    rad_sensor_angle : float
        The sensor angle in RADIANS (0 = right, π/2 = up, π = left, 3π/2 = down)
    max_range : float
        The maximum sensor range. Returns this value if no wall is detected.
    
    Returns
    -------
    float
        The distance to the closest wall in the sensor direction, capped at max_range.
        Returns max_range if no wall intersection is found within bounds. 
    """
    # Environment boundaries
    left_wall, right_wall = 0, 100
    bottom_wall, top_wall = 0, 100
    
    # Sensor direction components
    dx = np.cos(rad_sensor_angle)
    dy = np.sin(rad_sensor_angle)
    
    # Calculate intersection distances with all four walls
    intersections = []
    
    # Check right wall (x = 100)
    if dx >= 0:                                         # Sensor pointing right
        if dx == 0:                                     # avoid diving by 0
            dx += 0.001                                 
        t = (right_wall - x) / dx                       # distance to reach x = 100
        if t >= 0:                                       # if robot is in front of the wall
            y_intersect = y + dy * t                    # Y coordinate
            if bottom_wall <= y_intersect <= top_wall:  # Y inside limits
                intersections.append(t)                 # valid intersections
    
    # Check left wall (x = 0)
    if dx < 0:  # Sensor pointing left
        t = (left_wall - x) / dx
        if t >= 0:
            y_intersect = y + dy * t
            if bottom_wall <= y_intersect <= top_wall:
                intersections.append(t)
    
    # Check top wall (y = 100)
    if dy >= 0:  # Sensor pointing up
        if dy == 0:         # avoid diving by 0
            dx += 0.001
        t = (top_wall - y) / dy
        if t >= 0:
            x_intersect = x + dx * t
            if left_wall <= x_intersect <= right_wall:
                intersections.append(t)
    
    # Check bottom wall (y = 0)
    if dy < 0:  # Sensor pointing down
        t = (bottom_wall - y) / dy
        if t >= 0:
            x_intersect = x + dx * t
            if left_wall <= x_intersect <= right_wall:
                intersections.append(t)
    
    # Return the closest wall intersection within range
    if intersections:
        closest_wall = min(intersections)    #closest wall
        return min(closest_wall, max_range)
    else:
        return max_range                    # There is no walls in that direction
    
    
def calculate_distances(x, y, angle):
    """
    Calculate distances to obstacles and walls in three sensor directions.
    
    This function simulates three ultrasonic sensors (left, center, right)
    mounted on the robot. It returns the closest distance
    detected by each sensor, considering both obstacles and environment walls.
    
    Parameters
    ----------
    x : float
        The current x-coordinate of the robot (0 to 100)
    y : float
        The current y-coordinate of the robot (0 to 100)  
    angle : float
        The robot's heading angle in DEGREES (0° = right, 90° = up, 180° = left, 270° = down)
    
    Returns
    -------
    tuple (float, float, float)
        A tuple containing three distances in units:
        - left_distance: Distance to closest obstacle/wall on left sensor (+45°)
        - center_distance: Distance to closest obstacle/wall on center sensor (0°)  
        - right_distance: Distance to closest obstacle/wall on right sensor (-45°)
    """
    # Sensor configuration
    sensor_angles = [angle + 45, angle, angle - 45]  # Left, Center, Right
    sensor_fov = np.radians(30)  # ±30° field of view


    distances = []  # Default to max range
    
    for i, sensor_angle in enumerate(sensor_angles):
        rad_angle = np.radians(sensor_angle)
        
        # 1. FIRST: Check for WALLS (environment boundaries)
        wall_distance = calculate_wall_distance(x, y, rad_angle, sensor_range)

        # 2. THEN: Check for OBSTACLES (random objects)
        obstacle_dist = sensor_range
        for obs_x, obs_y in obstacles:
            # Vector from robot to obstacle
            dx = obs_x - x
            dy = obs_y - y
            distance_to_obs = np.sqrt(dx**2 + dy**2)
            
            # Skip if obstacle is beyond sensor range
            if distance_to_obs > sensor_range:
                continue
                
            # Calculate angle between sensor direction and obstacle direction
            obstacle_angle = np.arctan2(dy, dx)
            angle_diff = obstacle_angle - rad_angle
            
            # Normalize angle difference to [-π, π]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            
            if abs(angle_diff) <= sensor_fov and distance_to_obs < obstacle_dist:
                # Obstacle is detected by this sensor
                obstacle_dist = distance_to_obs
        
        # 3. Take the closer of wall or obstacle
        min_distance = min(wall_distance, obstacle_dist)
        distances.append(min_distance)
    
    return distances[0], distances[1], distances[2]  # left, center, right


# Helper function to plot fuzzy output
def plot_fuzzy_output(sim, consequent_name, ax):
    """
    Visualize the fuzzy logic output calculation for a specific consequent variable.
    
    This function creates a detailed visualization of the fuzzy inference process,
    showing membership functions, rule activations, aggregated output, and the
    final defuzzified value. It's essential for understanding and debugging the
    fuzzy logic decision-making process.
    
    Parameters
    ----------
    sim : ControlSystemSimulation
        The fuzzy logic simulation object containing the current state,
        rule activations, and computed outputs. Must have already called
        `compute()` method.
    consequent_name : str
        Name of the output variable to visualize (e.g., 'turn_angle').
        Must match a consequent defined in the control system.
    ax : matplotlib.axes.Axes
        The matplotlib axes object where the visualization will be drawn.
        The function will clear this axes before plotting.
    
    Returns
    -------
    None
        The function modifies the provided axes object in-place.
    """
    ax.clear()

    # Convert generator to dict {name: consequent}
    consequents = {c.label: c for c in sim.ctrl.consequents}
    if consequent_name not in consequents:
        raise ValueError(
            f"Consequent '{consequent_name}' not found. "
            f"Available: {list(consequents.keys())}"
        )

    consequent = consequents[consequent_name]

    # Plot all membership functions manually
    for term_name, mf in consequent.terms.items():
        ax.plot(
            consequent.universe,
            mf.mf, 
            label=str(term_name)
        )

    # Plot crisp (defuzzified) value if available
    if consequent_name in sim.output:
        defuzz_value = sim.output[consequent_name]
        ax.axvline(
            x=defuzz_value,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Decision: {defuzz_value:.2f}"
        )

    ax.set_title(f"Fuzzy Output: {consequent_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Universe")
    ax.set_ylabel("Membership Degree")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


# Input variables to the Fuzzy System
# Creates a numerical range that defines all possible values for our variable
# This represents all possible distance values of the sensor can detect (0-40 units), in steps of 1
left_distance = ctrl.Antecedent(np.arange(0, sensor_range+1, 1), 'left_distance')
center_distance = ctrl.Antecedent(np.arange(0, sensor_range+1, 1), 'center_distance')
right_distance = ctrl.Antecedent(np.arange(0, sensor_range+1, 1), 'right_distance')

# Output variable from the Fuzzy System
# Range from -90° to 90° (How much to turn left/right/straight)
turn_angle = ctrl.Consequent(np.arange(-90, 91, 1), 'turn_angle')

# Auto-membership function population (3, 5, or 7)
left_distance.automf(3,   names=['close', 'medium', 'far'])
center_distance.automf(3, names=['close', 'medium', 'far'])
right_distance.automf(3,  names=['close', 'medium', 'far'])

# Custom membership functions for turn angle
# Creates a triangular-shaped membership function
# Parameters: fuzz.trimf(universe, [a, b, c])
# a: Left corner (where membership begins)
# b: Peak (where membership = 1.0)
# c: Right corner (where membership ends)
turn_angle['sharp_left'] = fuzz.trimf(turn_angle.universe, [45, 90, 90])
turn_angle['left'] = fuzz.trimf(turn_angle.universe, [0, 30, 60])
turn_angle['straight'] = fuzz.trimf(turn_angle.universe, [-15, 0, 15])
turn_angle['right'] = fuzz.trimf(turn_angle.universe, [-60, -30, 0])
turn_angle['sharp_right'] = fuzz.trimf(turn_angle.universe, [-90, -90, -45])

#############################################
# Generating rules
#############################################
rules = [] #create a list
category = ['close', 'medium', 'far'] #create a list with the categories
for i, center in enumerate(category):
    for j, left in enumerate(category):
        for k, right in enumerate(category):
            # Determine appropriate action based on the combination
            if center == 'close':
                if left == 'close' and right == 'close':
                    action = 'sharp_left'  # All sides blocked, escape left
                elif left == 'close' and right == 'medium':
                    action = 'sharp_right'       # Right is somewhat clearer
                elif left == 'close' and right == 'far':
                    action = 'sharp_right'       # Right is clear
                elif left == 'medium' and right == 'close':
                    action = 'sharp_left'        # Left is somewhat clearer
                elif left == 'medium' and right == 'medium':
                    action = 'sharp_left'        # Both sides medium, choose left
                elif left == 'medium' and right == 'far':
                    action = 'sharp_right'       # Right is clearer
                elif left == 'far' and right == 'close':
                    action = 'sharp_left'        # Left is clear
                elif left == 'far' and right == 'medium':
                    action = 'sharp_left'        # Left is clearer
                elif left == 'far' and right == 'far':
                    action = 'sharp_left'        # Both sides clear, choose left
            
            elif center == 'medium':
                if left == 'close' and right == 'close':
                    action = 'straight'     # Narrow path, go straight carefully
                elif left == 'close' and right == 'medium':
                    action = 'right'        # Keep course with caution
                elif left == 'close' and right == 'far':
                    action = 'sharp_right'       # Favor clearer right side
                elif left == 'medium' and right == 'close':
                    action = 'left'    # Keep course with caution
                elif left == 'medium' and right == 'medium':
                    action = 'straight'    # Balanced situation, maintain course
                elif left == 'medium' and right == 'far':
                    action = 'right'    # Slight right favor, but straight
                elif left == 'far' and right == 'close':
                    action = 'sharp_left'        # Favor clearer left side
                elif left == 'far' and right == 'medium':
                    action = 'sharp_left'    # Slight left favor, but straight
                elif left == 'far' and right == 'far':
                    action = 'sharp_right'    # go to a clearer path
            
            else:  # center == 'far'
                if left == 'close' and right == 'close':
                    action = 'straight'    # Narrow but clear path ahead
                elif left == 'close' and right == 'medium':
                    action = 'straight'    # Clear path ahead
                elif left == 'close' and right == 'far':
                    action = 'straight'    # Clear path ahead
                elif left == 'medium' and right == 'close':
                    action = 'straight'    # Clear path ahead
                elif left == 'medium' and right == 'medium':
                    action = 'straight'    # Perfectly clear
                elif left == 'medium' and right == 'far':
                    action = 'straight'    # Clear path ahead
                elif left == 'far' and right == 'close':
                    action = 'straight'    # Clear path ahead
                elif left == 'far' and right == 'medium':
                    action = 'straight'    # Clear path ahead
                elif left == 'far' and right == 'far':
                    action = 'straight'    # Completely clear, full speed ahead
            
            # Create the rule
            rule = ctrl.Rule(center_distance[center] & left_distance[left] & right_distance[right], turn_angle[action])
            rules.append(rule)

print(f"Total rules geneated: {len(rules)}/27")

# Create control system, fuzzy objects
navigation_ctrl = ctrl.ControlSystem(rules)  #Creates the fuzzy logic engine of the system.
navigation_system = ctrl.ControlSystemSimulation(navigation_ctrl)  #Creates a simulation instance for making specific calculations


print("Fuzzy control system created successfully!")


# Simulation setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.3)  # Add spacing between subplots

# Environment boundaries
map_width, map_height = 100, 100

# Robot and environment parameters
robot_x, robot_y = 15, 15 #initial position of the robot
robot_angle = 45  # Start pointing right
robot_size = 2   # Visual size of robot

goal_x, goal_y = 85, 85
goal = (goal_x, goal_y)   # Goal in top-right corner
goal_size = 5     # Visual size of goal

# Create obstacles avoiding start and goal areas
obstacles = []
obstacle_size = 5
number_of_obstacles = 10
while len(obstacles) < number_of_obstacles:
    obs_x = random.randint(20, 80)
    obs_y = random.randint(20, 80)
    
    # Check distance from start position
    dist_from_start = np.sqrt((obs_x - robot_x)**2 + (obs_y - robot_y)**2)
    # Check distance from goal
    dist_from_goal = np.sqrt((obs_x - goal_x)**2 + (obs_y - goal_y)**2)
    
    if dist_from_start > 25 and dist_from_goal > 25:
        obstacles.append((obs_x, obs_y))


def update(frame):
    """
    Update function for the robot navigation animation.
    
    This function is called repeatedly by matplotlib's FuncAnimation to update
    each frame of the simulation. It handles robot movement, sensor readings,
    fuzzy logic decision-making, and visualization updates.
    
    Parameters
    ----------
    frame : int
        The current frame number of the animation. Used for display purposes
        and to control animation timing.
    
    Returns
    -------
    tuple of matplotlib.axes.Axes
        Returns (ax1, ax2) - the two axes objects containing the updated plots.
        This return is required by FuncAnimation to refresh the display.

    """
    global robot_x, robot_y, robot_angle

    sensor_angles = [45,0,-45]
    
    # Calculate distances to obstacles using sensors
    left_dist, center_dist, right_dist = calculate_distances(robot_x, robot_y, robot_angle)
    
    # Set inputs to fuzzy system
    navigation_system.input['left_distance'] = left_dist
    navigation_system.input['center_distance'] = center_dist
    navigation_system.input['right_distance'] = right_dist
    
    # Compute the output using fuzzy logic
    navigation_system.compute()
    turn_angle_output  = navigation_system.output['turn_angle']
    
    # Update robot position and orientation
    robot_angle += turn_angle_output * 0.1 #Apply turn with damping factor
    rad_angle = np.radians(robot_angle)
    robot_x += np.cos(rad_angle) * 2
    robot_y += np.sin(rad_angle) * 2
    
    # Boundary checking
    robot_x = max(0, min(100, robot_x))
    robot_y = max(0, min(100, robot_y))
    
    # Clear previous plot elements
    ax1.clear()
    ax2.clear()
    
    # Plot obstacles
    for obs_x, obs_y in obstacles:
        ax1.add_patch(plt.Circle((obs_x, obs_y), obstacle_size, color='red', alpha=0.7))
    
    # Plot goal
    ax1.add_patch(plt.Circle(goal, goal_size, color='green', alpha=0.7, label = 'Goal'))
    
    # Plot robot
    ax1.add_patch(plt.Circle((robot_x, robot_y), robot_size, color='blue', label = 'robot'))
    
    # Plot robot direction
    dx = np.cos(np.radians(robot_angle)) * 10
    dy = np.sin(np.radians(robot_angle)) * 10
    ax1.arrow(robot_x, robot_y, dx, dy, head_width=2, head_length=3, fc='blue', ec='blue', width = 1.2)
    

    # Plot sensor ranges
    sensor_colors = ['orange', 'yellow', 'cyan']
    sensor_labels = ['Left Sensor', 'Center Sensor', 'Right Sensor']
    
    for i, sensor_offset in enumerate(sensor_angles):
        sensor_angle = robot_angle + sensor_offset
        rad_angle = np.radians(sensor_angle)
        end_x = robot_x + np.cos(rad_angle) * sensor_range
        end_y = robot_y + np.sin(rad_angle) * sensor_range

        ax1.plot([robot_x, end_x], [robot_y, end_y], color=sensor_colors[i], 
                linestyle='--', alpha=0.5, label=sensor_labels[i] )
    
    # Configure main plot
    ax1.set_xlim(0, map_width)
    ax1.set_ylim(0, map_height)
    ax1.set_title('Fuzzy Logic Robot Navigation\nFrame: {}'.format(frame))
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper center')
    
    # Display sensor readings and decision information
    info_text = (
        f'Sensor Readings:\n'
        f'Left: {left_dist:.1f} units\n'
        f'Center: {center_dist:.1f} units\n' 
        f'Right: {right_dist:.1f} units\n\n'
        f'Fuzzy Decision:\n'
        f'Turn: {turn_angle_output:.1f}°\n'
        f'Heading: {robot_angle:.1f}°\n\n'
        f'Position: ({robot_x:.1f}, {robot_y:.1f})'
    )
    
    ax1.text(5, 95, info_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Draw fuzzy logic output visualization
    plot_fuzzy_output(navigation_system, 'turn_angle', ax2)
    
    # Calculate distance to goal
    goal_distance = np.sqrt((robot_x - goal_x)**2 + (robot_y - goal_y)**2)
    
    # Check if goal is reached
    if goal_distance < 7:
        ax1.text(robot_x, robot_y + 12, "GOAL REACHED!", fontsize=14, 
                color='green', weight='bold', ha='center',
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
        ani.event_source.stop()
        return ax1, ax2
    
    # Check for collision with obstacles
    for obs_x, obs_y in obstacles:
        obs_distance = np.sqrt((robot_x - obs_x)**2 + (robot_y - obs_y)**2)
        if obs_distance < 5:
            ax1.text(robot_x, robot_y - 12, "COLLISION!", fontsize=14,
                    color='red', weight='bold', ha='center',
                    bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
            ani.event_source.stop()
            return ax1, ax2
        
    # Display progress towards goal
    progress = 100 * (1 - goal_distance / np.sqrt((goal_x - 15)**2 + (goal_y - 15)**2))
    ax1.text(goal_x, goal_y + 8, f"Goal: {goal_distance:.1f} units\nProgress: {progress:.1f}%", 
             fontsize=8, ha='center', color='darkgreen',
             bbox=dict(facecolor='lightgreen', alpha=0.7))
    
    return ax1, ax2

# Create animation
ani = FuncAnimation(fig, update, frames=500, interval=200, repeat=False, blit=False) #update will be called 200 times, 200 ms per frame
plt.tight_layout()
plt.show()