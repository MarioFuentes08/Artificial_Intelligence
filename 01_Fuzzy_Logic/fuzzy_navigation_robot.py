import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


def validate_navigation_rules_coverage(ctrl_system):
    """Validate that all combinatios are covered"""

    terms = ['close', 'medium', 'far']
    all_combinations = set()
    
    # Create a set
    covered_combinations = set() 

    # Generate all possible combinations (3^3 = 27)
    for center in terms:
        for left in terms:
            for right in terms:
                all_combinations.add((center, left, right))
     
    # Analyze covered combinations by the existing rules
    for rule in ctrl_system.rules:
        # Extract the rules terms
        rule_str = str(rule).lower()
        terms_in_rule = ['temp','temp','temp']

        
        close_term_count  = rule_str.count(terms[0])
        medium_term_count = rule_str.count(terms[1])
        far_term_count    = rule_str.count(terms[2])

        close_term_index = []
        medium_term_index = []
        far_term_index = []

        is_close_term = rule_str.find(terms[0])
        while is_close_term != -1:
            close_term_index.append(is_close_term)
            is_close_term = rule_str.find(terms[0], is_close_term + 1)


        is_medium_term = rule_str.find(terms[1])
        while is_medium_term != -1:
            medium_term_index.append(is_medium_term)
            is_medium_term = rule_str.find(terms[1], is_medium_term + 1)


        is_far_term = rule_str.find(terms[2])
        while is_far_term != -1:
            far_term_index.append(is_far_term)
            is_far_term = rule_str.find(terms[2], is_far_term + 1)
      
        for i in range(close_term_count):
            terms_in_rule.append(terms[0])
        for j in range(medium_term_count):
            terms_in_rule.append(terms[1])
        for z in range(far_term_count):
            terms_in_rule.append(terms[2])

        if len(terms_in_rule) == 3:
            covered_combinations.add(tuple(terms_in_rule))        
    # Find missing combinations
    missing = all_combinations - covered_combinations
    return missing

# Input variables to the Fuzzy System
# Creates a numerical range that defines all possible values for our variable
# This represents all possible distance values of the sensor can detect (0-100 units), in steps of 1
left_distance = ctrl.Antecedent(np.arange(0, 101, 1), 'left_distance')
center_distance = ctrl.Antecedent(np.arange(0, 101, 1), 'center_distance')
right_distance = ctrl.Antecedent(np.arange(0, 101, 1), 'right_distance')

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
turn_angle['sharp_left'] = fuzz.trimf(turn_angle.universe, [-90, -90, -45])
turn_angle['left'] = fuzz.trimf(turn_angle.universe, [-60, -30, 0])
turn_angle['straight'] = fuzz.trimf(turn_angle.universe, [-15, 0, 15])
turn_angle['right'] = fuzz.trimf(turn_angle.universe, [0, 30, 60])
turn_angle['sharp_right'] = fuzz.trimf(turn_angle.universe, [45, 90, 90])

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
                    action = 'right'       # Right is somewhat clearer
                elif left == 'close' and right == 'far':
                    action = 'sharp_right'       # Right is clear
                elif left == 'medium' and right == 'close':
                    action = 'left'        # Left is somewhat clearer
                elif left == 'medium' and right == 'medium':
                    action = 'left'        # Both sides medium, choose left
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
                    action = 'straight'    # Narrow path, go straight carefully
                elif left == 'close' and right == 'medium':
                    action = 'straight'    # Keep course with caution
                elif left == 'close' and right == 'far':
                    action = 'sharp_right'       # Favor clearer right side
                elif left == 'medium' and right == 'close':
                    action = 'straight'    # Keep course with caution
                elif left == 'medium' and right == 'medium':
                    action = 'straight'    # Balanced situation, maintain course
                elif left == 'medium' and right == 'far':
                    action = 'straight'    # Slight right favor, but straight
                elif left == 'far' and right == 'close':
                    action = 'sharp_left'        # Favor clearer left side
                elif left == 'far' and right == 'medium':
                    action = 'sharp_left'    # Slight left favor, but straight
                elif left == 'far' and right == 'far':
                    action = 'sharp_right'    # Path opening up, maintain course
            
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

# Verify whether there is no missing rules, if so, stop execution
print("=" * 20)
print("Verify rules")
print("=" * 20)
#missing_rule = validate_navigation_rules_coverage(navigation_ctrl)
#print(f"Missing combinations: {missing_rule} \n")
#print(f"There are {len(missing_rule)} missing combinations")
#exit()

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
number_of_obstacles = 5
while len(obstacles) < number_of_obstacles:
    obs_x = random.randint(20, 80)
    obs_y = random.randint(20, 80)
    
    # Check distance from start position
    dist_from_start = np.sqrt((obs_x - robot_x)**2 + (obs_y - robot_y)**2)
    # Check distance from goal
    dist_from_goal = np.sqrt((obs_x - goal_x)**2 + (obs_y - goal_y)**2)
    
    if dist_from_start > 25 and dist_from_goal > 25:
        obstacles.append((obs_x, obs_y))



# Sensor ranges
sensor_range = 40

def calculate_wall_distance(x, y, sensor_angle, max_range):
    """
    Calculate distance to the nearest wall in the sensor direction
    """
    # Environment boundaries
    left_wall, right_wall = 0, 100
    bottom_wall, top_wall = 0, 100
    
    # Sensor direction components
    dx = np.cos(sensor_angle)
    dy = np.sin(sensor_angle)
    
    # Calculate intersection distances with all four walls
    intersections = []
    
    # Check right wall (x = 100)
    if dx > 0:  # Sensor pointing right
        t = (right_wall - x) / dx
        if t > 0:
            y_intersect = y + dy * t
            if bottom_wall <= y_intersect <= top_wall:
                intersections.append(t)
    
    # Check left wall (x = 0)
    if dx < 0:  # Sensor pointing left
        t = (left_wall - x) / dx
        if t > 0:
            y_intersect = y + dy * t
            if bottom_wall <= y_intersect <= top_wall:
                intersections.append(t)
    
    # Check top wall (y = 100)
    if dy > 0:  # Sensor pointing up
        t = (top_wall - y) / dy
        if t > 0:
            x_intersect = x + dx * t
            if left_wall <= x_intersect <= right_wall:
                intersections.append(t)
    
    # Check bottom wall (y = 0)
    if dy < 0:  # Sensor pointing down
        t = (bottom_wall - y) / dy
        if t > 0:
            x_intersect = x + dx * t
            if left_wall <= x_intersect <= right_wall:
                intersections.append(t)
    
    # Return the closest wall intersection within range
    if intersections:
        closest_wall = min(intersections)
        return min(closest_wall, max_range)
    else:
        return max_range
    
    
def calculate_distances(x, y, angle):
    """
    Calculate distances to obstacles in three directions: left, center, right
    """
    # Sensor configuration
    sensor_angles = [angle + 45, angle, angle - 45]  # Left, Center, Right
    sensor_range = 40
    sensor_fov = np.radians(30)  # ±30° field of view


    distances = [sensor_range, sensor_range, sensor_range]  # Default to max range
    
    for i, sensor_angle in enumerate(sensor_angles):
        min_distance = sensor_range
        rad_angle = np.radians(sensor_angle)
        
        # 1. FIRST: Check for WALLS (environment boundaries)
        wall_distance = calculate_wall_distance(x, y, rad_angle, sensor_range)
        if wall_distance < min_distance:
            min_distance = wall_distance

        # 2. THEN: Check for OBSTACLES (random objects)
        
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
            
            if abs(angle_diff) <= sensor_fov:
                # Obstacle is detected by this sensor
                if distance_to_obs < min_distance:
                    min_distance = distance_to_obs
        
        distances[i] = min_distance
    
    return distances[0], distances[1], distances[2]  # left, center, right


# Helper function to plot fuzzy output
def plot_fuzzy_output(sim, consequent_name, ax):
    """
    Plot fuzzy output with membership functions and defuzzified decision line.
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



def update(frame):
    global robot_x, robot_y, robot_angle

    sensor_angles = [-45,0,45]
    
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
        if obs_distance < 7:
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