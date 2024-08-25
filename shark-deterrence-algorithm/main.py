import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from scipy.spatial.distance import euclidean

# Constants
OCEAN_WIDTH = 10
OCEAN_HEIGHT = 10
FRAME_INTERVAL = 100
NUM_FRAMES = 1000
NUM_SHARKS = 5

# Speeds in km/h, converting them to the simulation units
PERSON_SPEED = 0.5 / 60
PERSON_SPEED_BOOST = 1.2 / 60
SHARK_SPEED = 1.8 / 60
SCARED_DURATION = 200

# Threshold distances
SOUND_TRIGGER_DISTANCE = 0.5
SPEED_BOOST_DISTANCE = 1.0
MIN_SHARK_DISTANCE = 0.5  # Minimum distance between sharks to prevent clustering

# Sound generation functions
def generate_sound_wave(frequency, duration, amplitude=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

def play_sound(wave_data, sample_rate=44100):
    try:
        sd.play(wave_data, sample_rate)
    except sd.PortAudioError as e:
        print(f"Error playing sound: {e}")

class Shark:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.speed = SHARK_SPEED
        self.scared = False
        self.scared_frames = 0
        self.direction = np.random.uniform(0, 2 * np.pi)

    def move(self, person_x, person_y, ocean):
        if self.scared:
            if self.scared_frames > 0:
                angle = self.get_escape_angle(person_x, person_y)
                self.scared_frames -= 1
            else:
                self.scared = False
                angle = np.arctan2(person_y - self.y, person_x - self.x)
        else:
            angle = np.arctan2(person_y - self.y, person_x - self.x)

        # Repulsion from other sharks
        for other_shark in ocean.sharks:
            if other_shark.id != self.id:
                distance_to_other_shark = euclidean((self.x, self.y), (other_shark.x, other_shark.y))
                if distance_to_other_shark < MIN_SHARK_DISTANCE:
                    angle_away = np.arctan2(self.y - other_shark.y, self.x - other_shark.x)
                    angle = 0.7 * angle + 0.3 * angle_away  # Blend the avoidance with the main direction

        dx = self.speed * np.cos(angle)
        dy = self.speed * np.sin(angle)

        # Update position with boundary handling
        self.x += dx
        self.y += dy

        # Reflect off boundaries to prevent unrealistic behavior
        if self.x <= 1 or self.x >= ocean.width - 1:
            dx *= -1  # Reverse direction on the x-axis
        if self.y <= 1 or self.y >= ocean.height - 1:
            dy *= -1  # Reverse direction on the y-axis

        self.x = np.clip(self.x, 1, ocean.width - 1)
        self.y = np.clip(self.y, 1, ocean.height - 1)

    def get_escape_angle(self, person_x, person_y):
        """Determine the escape angle based on the chosen escape behavior."""
        return np.arctan2(self.y - person_y, self.x - person_x) + np.pi / 2  # Circular escape

    def scare_away(self):
        self.scared = True
        self.scared_frames = SCARED_DURATION
        self.speed = SHARK_SPEED * 1.5  # Increase speed when scared

class Person:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.base_speed = PERSON_SPEED
        self.speed = self.base_speed
        self.reached_ship = False

    def move(self, shark_positions, ship_x, ship_y, ocean):
        if self.reached_ship:
            return  # Stop moving if the person has reached the ship

        # Find the closest shark
        closest_shark_distance = min(
            [euclidean((self.x, self.y), (shark_x, shark_y)) for shark_x, shark_y in shark_positions])

        # If the closest shark is within the boost distance, increase speed significantly
        if closest_shark_distance < SPEED_BOOST_DISTANCE:
            self.speed = PERSON_SPEED_BOOST * 1.5  # Increased speed boost to 1 km/h
        else:
            self.speed = self.base_speed

        # Move towards the ship
        angle = np.arctan2(ship_y - self.y, ship_x - self.x)
        dx = self.speed * np.cos(angle)
        dy = self.speed * np.sin(angle)
        self.x = np.clip(self.x + dx, 1, ocean.width - 1)
        self.y = np.clip(self.y + dy, 1, ocean.height - 1)

        # Check if the person reached the ship
        if euclidean((self.x, self.y), (ship_x, ship_y)) < 0.1:  # Within 100 meters of the ship
            self.reached_ship = True
            ocean.log_event("The person has reached the ship! Generating animation...")

class Ocean:
    def __init__(self, width, height, num_sharks):
        self.width = width
        self.height = height
        self.sharks = [Shark(np.random.uniform(1, width - 1), np.random.uniform(1, height - 1), id=i) for i in
                       range(num_sharks)]
        self.ship_x, self.ship_y = self.place_ship_away_from_boundaries()
        self.person = self.place_person_far_from_ship()  # Start the person far from the ship
        self.sound_waves = []  # List of active sound waves
        self.logs = []  # Logs of important events

    def place_ship_away_from_boundaries(self):
        """Place the ship away from the corners and boundaries of the ocean."""
        margin = 1  # Margin to keep the ship away from the edges
        x = np.random.uniform(margin, self.width - margin)
        y = np.random.uniform(margin, self.height - margin)
        return x, y

    def place_person_far_from_ship(self):
        """Place the person far away from the ship to test the effectiveness of the simulation."""
        max_distance = max(self.width, self.height) * 0.8  # Person should be at 80% of the maximum possible distance
        person_x = np.clip(self.ship_x + max_distance * np.cos(np.random.uniform(0, 2 * np.pi)), 1, self.width - 1)
        person_y = np.clip(self.ship_y + max_distance * np.sin(np.random.uniform(0, 2 * np.pi)), 1, self.height - 1)
        return Person(person_x, person_y)

    def log_event(self, event):
        self.logs.append(event)
        print(event)

    def move_all(self):
        if self.person.reached_ship:
            return False  # Stop simulation movement when the person reaches the ship

        # Move the sharks
        for shark in self.sharks:
            shark.move(self.person.x, self.person.y, self)

        # Calculate distances
        shark_positions = [(shark.x, shark.y) for shark in self.sharks]

        # If any shark is too close, increase person's speed and trigger the sound
        for shark in self.sharks:
            distance_to_shark = euclidean((self.person.x, self.person.y), (shark.x, shark.y))
            if distance_to_shark < SOUND_TRIGGER_DISTANCE and not shark.scared:
                deterrent_sound = generate_sound_wave(350, duration=0.5)  # Changed frequency for a new tone
                play_sound(deterrent_sound)
                shark.scare_away()
                self.sound_waves.append({'radius': 0.05, 'center': (self.person.x, self.person.y)})  # Add a new sound wave
                self.log_event(f"Shark {shark.id + 1} is close to the person. Sound effect started.")

        # Move the person towards the ship
        self.person.move(shark_positions, self.ship_x, self.ship_y, self)

        # Update the sound waves
        for wave in self.sound_waves:
            wave['radius'] += 1  # Increase the radius of each wave by 1 unit
        self.sound_waves = [wave for wave in self.sound_waves if wave['radius'] <= OCEAN_WIDTH]  # Remove waves that exceed the ocean size

        return True  # Continue the simulation

# Initialize ocean
ocean = Ocean(OCEAN_WIDTH, OCEAN_HEIGHT, NUM_SHARKS)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Define colors
shark_color = 'red'
person_color = 'blue'
ship_color = 'green'

shark_scatters = [ax.scatter([shark.x], [shark.y], c=shark_color, marker='s', s=50) for shark in ocean.sharks]
person_scatter = ax.scatter([ocean.person.x], [ocean.person.y], c=person_color, marker='o', s=50)
ship_scatter = ax.scatter([ocean.ship_x], [ocean.ship_y], c=ship_color, marker='^', s=50)

# Labels
person_label = ax.text(ocean.person.x, ocean.person.y - 0.3, 'Person', fontsize=10, ha='center', color=person_color,
                       weight='bold')
ship_label = ax.text(ocean.ship_x, ocean.ship_y - 0.3, 'Ship', fontsize=10, ha='center', color=ship_color, weight='bold')
shark_labels = [
    ax.text(shark.x, shark.y - 0.3, f'S {shark.id + 1}', fontsize=10, ha='center', color=shark_color, weight='bold')
    for shark in ocean.sharks]

# Sound waves plot (initially empty)
sound_wave_circles = [plt.Circle((0, 0), radius=0, color='cyan', fill=False, lw=2) for _ in range(NUM_SHARKS * 2)]
for circle in sound_wave_circles:
    ax.add_patch(circle)

ax.set_xlim(0, ocean.width)
ax.set_ylim(0, ocean.height)
ax.set_title('Shark and Person Escape Simulation')

# Text annotations
distance_to_shark_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top', fontsize=7,
                                 weight='bold')
distance_to_ship_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, verticalalignment='top', fontsize=7,
                                weight='bold')

def update(frame):
    continue_simulation = ocean.move_all()

    for scatter, shark in zip(shark_scatters, ocean.sharks):
        scatter.set_offsets([(shark.x, shark.y)])

    person_scatter.set_offsets([(ocean.person.x, ocean.person.y)])
    ship_scatter.set_offsets([(ocean.ship_x, ocean.ship_y)])

    # Update labels
    person_label.set_position((ocean.person.x, ocean.person.y - 0.3))
    ship_label.set_position((ocean.ship_x, ocean.ship_y - 0.3))
    for label, shark in zip(shark_labels, ocean.sharks):
        label.set_position((shark.x, shark.y - 0.3))

    # Update sound waves
    for i, wave in enumerate(ocean.sound_waves):
        wave['radius'] += 0.2  # Increase the radius of each wave by 0.2 units
        sound_wave_circles[i].center = wave['center']
        sound_wave_circles[i].set_radius(wave['radius'])

    # Hide remaining circles if not used
    for i in range(len(ocean.sound_waves), len(sound_wave_circles)):
        sound_wave_circles[i].set_radius(0)

    # Calculate distances
    distance_to_shark = min([euclidean((ocean.person.x, ocean.person.y), (shark.x, shark.y)) for shark in ocean.sharks])
    distance_to_ship = euclidean((ocean.person.x, ocean.person.y), (ocean.ship_x, ocean.ship_y))

    # Update distance texts
    distance_to_shark_text.set_text(f'Distance to closest shark: {distance_to_shark:.2f} units')
    distance_to_ship_text.set_text(f'Distance to ship: {distance_to_ship:.2f} units')

    if not continue_simulation:
        return [*shark_scatters, person_scatter, ship_scatter, person_label, ship_label, *shark_labels,
                distance_to_shark_text, distance_to_ship_text, *sound_wave_circles]

    return [*shark_scatters, person_scatter, ship_scatter, person_label, ship_label, *shark_labels,
            distance_to_shark_text, distance_to_ship_text, *sound_wave_circles]

anim = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=FRAME_INTERVAL, blit=True)
plt.show()

print("Simulation complete.")