# Hybrid GA + per-agent LSTM policies - OPTIMIZED with FIXED CHECKPOINTS
import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pygame
import random
import math
import time
import matplotlib.pyplot as plt




pygame.init()




# Window
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Hybrid GA + LSTM AIs")
clock = pygame.time.Clock()




# Font
pygame.font.init()
font = pygame.font.SysFont(None, 24)




# Track parameters
track_width = 60




START_X = 340
START_Y = 395




# Simulation params
num_ais = 50
dot_speed = 3
generation_length = 2000  # frames per generation
current_frame = 0




# Hybrid hyperparams (epsilon-greedy exploration)
epsilon = 0.9           # Start at 90% random exploration
epsilon_decay = 0.995   # Multiply epsilon by this each generation
epsilon_min = 0.1       # Minimum exploration rate (always keep 10% random)
seq_len = 8                  # how many past timesteps to feed the LSTM
train_epochs_per_agent = 2   # REDUCED from 5 - fewer epochs = faster training
batch_size = 64              # INCREASED from 32 - larger batches = faster training
mutation_std = 0.02          # std dev for Gaussian weight mutation applied to children
learning_rate = 0.001
train_every_n_generations = 1  # Train every N generations (set to 2-3 to train less often)


# === LIVE GRAPH TRACKING ===
gen_history = []
score_history = []


plt.ion()  # interactive mode ON
fig, ax = plt.subplots()
scatter_plot = ax.scatter([], [])
line_plot, = ax.plot([], [], '-')  # line connecting scatter points


ax.set_xlabel("Generation")
ax.set_ylabel("Best Food Score")
ax.set_title("Evolution Performance Over Generations")




# NEW: Experience replay buffer - store best experiences across generations
replay_buffer_size = 5000
global_replay_buffer = np.zeros((0, 7), dtype=float)  # shared good experiences




# Per-agent memory: list of arrays, each row: [x, y, onTrack, total_progress, lap, food, action]
ai_memory = [np.zeros((0, 7), dtype=float) for _ in range(num_ais)]
food_penalty_array = np.zeros(num_ais)  # penalty array




# Per-agent state
dot_positions = np.array([[START_X, START_Y] for _ in range(num_ais)], dtype=float)
laps = np.zeros(num_ais, dtype=int)  # Start at 0
max_laps = np.zeros(num_ais, dtype=int)  # Track highest lap achieved (prevents exploit)
food_array = np.zeros(num_ais)
finished_last_frame = np.zeros(num_ais, dtype=bool)
is_alive = np.ones(num_ais, dtype=bool)  # Track which AIs are alive
off_track_count = np.zeros(num_ais, dtype=int)  # Count consecutive off-track moves
max_off_track = 10  # Maximum allowed off-track moves before death
agent_epsilons = np.full(num_ais, epsilon)  # Per-agent exploration rates
parent_death_positions = {}  # dictionary: {parent_idx: (x, y)}




# NEW CHECKPOINT SYSTEM
# Five checkpoints: bottom-left curve, top-left curve, top-right curve, bottom-right curve, finish
checkpoints = [
    np.array([200, 400], dtype=float),  # CP 0: bottom-left curve (start area)
    np.array([140, 380], dtype=float),  # CP 1: left side curve
    np.array([110, 315], dtype=float),  # CP 2: left side curve
    np.array([125, 250], dtype=float),  # CP 3: top-left curve
    np.array([180, 215], dtype=float),  # CP 4: top-left curve
    np.array([250, 200], dtype=float),  # CP 5: top-left curve
    np.array([350, 200], dtype=float),  # CP 6: top-left curve
    np.array([600, 200], dtype=float),  # CP 7: top-right curve
    np.array([600, 400], dtype=float),  # CP 8: bottom-right curve
]
num_checkpoints = len(checkpoints)
checkpoint_radius = 30  # radius for checkpoint "hit"
checkpoints_reached = np.zeros(num_ais, dtype=int)  # How many checkpoints passed in current lap
checkpoint_flags = np.zeros((num_ais, num_checkpoints), dtype=bool)  # Track which checkpoints hit this lap




# Track geometry
def generate_track_points():
    outer_points = []
    inner_points = []
    for angle in range(90, 271, 2):
        rad = math.radians(angle)
        x_out = 200 + 120 * math.cos(rad)
        y_out = 300 + 125 * math.sin(rad)
        outer_points.append((x_out, y_out))
        x_in = 200 + (120 - track_width) * math.cos(rad)
        y_in = 300 + (125 - track_width) * math.sin(rad)
        inner_points.append((x_in, y_in))
    outer_points.append((600, 175))
    inner_points.append((600, 175 + track_width))
    for angle in range(-90, 91, 2):
        rad = math.radians(angle)
        x_out = 600 + 120 * math.cos(rad)
        y_out = 300 + 125 * math.sin(rad)
        outer_points.append((x_out, y_out))
        x_in = 600 + (120 - track_width) * math.cos(rad)
        y_in = 300 + (125 - track_width) * math.sin(rad)
        inner_points.append((x_in, y_in))
    outer_points.append((200, 425))
    inner_points.append((200, 425 - track_width))
    return outer_points, inner_points




outer_boundary, inner_boundary = generate_track_points()




def is_on_track(x, y):
    # left semicircle
    if x < 400:
        dist = math.hypot(x - 200, y - 300)
        outer_radius = 120
        inner_radius = 120 - track_width
        if inner_radius <= dist <= outer_radius:
            angle = math.degrees(math.atan2(y - 300, x - 200))
            if 90 <= angle <= 270 or angle <= -90:
                return True
    else:
        dist = math.hypot(x - 600, y - 300)
        outer_radius = 120
        inner_radius = 120 - track_width
        if inner_radius <= dist <= outer_radius:
            angle = math.degrees(math.atan2(y - 300, x - 600))
            if -90 <= angle <= 90:
                return True
    if 200 <= x <= 600 and 175 <= y <= 175 + track_width:
        return True
    if 200 <= x <= 600 and 425 - track_width <= y <= 425:
        return True
    return False




def is_in_finish_zone(x, y):
    """Check if position is in the finish zone (bottom straight near start)"""
    return 380 <= x <= 420 and 365 <= y <= 425




def update_memory(ai_idx, x, y, onTrack, total_progress, lap, food, action):
    step = np.array([[x, y, float(onTrack), total_progress, float(lap), food, float(action)]], dtype=float)
    ai_memory[ai_idx] = np.concatenate((ai_memory[ai_idx], step), axis=0)




# Model factory (PRE-COMPILE to avoid lag) - SMALLER for faster inference
def create_model(compile_model=True):
    m = keras.Sequential([
        layers.LSTM(32, input_shape=(None, 6), return_sequences=False),  # Reduced from 64
        layers.Dense(16, activation='relu'),  # Reduced from 32
        layers.Dense(4, activation='softmax')
    ])
    if compile_model:
        m.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return m




# Per-agent models - PRE-BUILD all models at start
models = [create_model() for _ in range(num_ais)]




# NEW: Cache for predictions to reduce overhead
prediction_cache = {}




# Helper: build training samples from ai_memory for one agent
def build_training_data(mem, seq_len=8, min_food_threshold=None):
    """
    mem: (T,7) with columns [x,y,onTrack,total_progress,lap,food,action]
    min_food_threshold: if set, only use experiences with food >= threshold
    returns X:(n, seq_len, 6) and y:(n,) actions
    """
    if mem.shape[0] < 2:
        return np.zeros((0, seq_len, 6)), np.zeros((0,), dtype=int)
   
    # Filter by food quality if threshold provided
    if min_food_threshold is not None:
        food_values = mem[:, 5]
        good_indices = np.where(food_values >= min_food_threshold)[0]
        if len(good_indices) < 2:
            return np.zeros((0, seq_len, 6)), np.zeros((0,), dtype=int)
        mem = mem[good_indices]
   
    states = mem[:, :6]  # (T,6)
    actions = mem[:, 6].astype(int)  # (T,)
    X_list = []
    y_list = []
    T = states.shape[0]
    for t in range(1, T):
        start = max(0, t - seq_len)
        window = states[start:t]
        if window.shape[0] < seq_len:
            pad = np.zeros((seq_len - window.shape[0], 6), dtype=float)
            window = np.vstack((pad, window))
        X_list.append(window)
        y_list.append(actions[t])
    if len(X_list) == 0:
        return np.zeros((0, seq_len, 6)), np.zeros((0,), dtype=int)
    return np.stack(X_list, axis=0), np.array(y_list, dtype=int)




# Helper: mutate weights (list of numpy arrays)
def mutate_weights(weights, std=0.02):
    new = []
    for w in weights:
        noise = np.random.normal(0.0, std, size=w.shape).astype(w.dtype)
        new.append(w + noise)
    return new




# NEW: Transfer weights without creating new model objects (ELIMINATES LAG)
def transfer_weights_mutated(target_model, source_weights, std=0.02):
    """Directly set mutated weights on existing model - much faster than creating new models"""
    mutated = mutate_weights(source_weights, std)
    target_model.set_weights(mutated)




# Hyper-sanity: ensure reproducibility if needed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)




generation = 0




running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False




    # --- Per-agent step ---
    # OPTIMIZATION: Batch all predictions together for speed
    batch_inputs = []
    batch_ai_indices = []
    actions = np.zeros(num_ais, dtype=int)
   
    for ai_idx in range(num_ais):
        # Skip dead AIs
        if not is_alive[ai_idx]:
            continue
       
        # Epsilon-greedy action selection - use per-agent epsilon
        current_epsilon = agent_epsilons[ai_idx]
        if np.random.rand() < current_epsilon:
            actions[ai_idx] = random.randint(0, 3)  # Explore: random action
        else:
            # Prepare last seq_len states for this agent
            mem = ai_memory[ai_idx]
            if mem.shape[0] == 0:
                actions[ai_idx] = random.randint(0, 3)
            else:
                # get last seq_len rows of states
                states = mem[:, :6]
                window = states[-seq_len:] if states.shape[0] >= seq_len else states
                if window.shape[0] < seq_len:
                    pad = np.zeros((seq_len - window.shape[0], 6), dtype=float)
                    window = np.vstack((pad, window))
               
                # Check cache first
                window_key = window.tobytes()
                if window_key in prediction_cache:
                    actions[ai_idx] = prediction_cache[window_key]
                else:
                    # Add to batch for prediction
                    batch_inputs.append(window)
                    batch_ai_indices.append(ai_idx)
   
    # Batch predict all at once
    if len(batch_inputs) > 0:
        batch_array = np.stack(batch_inputs, axis=0)  # Shape: (batch_size, seq_len, 6)
       
        # Predict with first model (all models should be similar after breeding)
        probs_batch = models[0].predict(batch_array, verbose=0)
       
        # Assign actions and cache results
        for i, ai_idx in enumerate(batch_ai_indices):
            action = int(np.argmax(probs_batch[i]))
            actions[ai_idx] = action
           
            # Cache this prediction
            window_key = batch_inputs[i].tobytes()
            prediction_cache[window_key] = action
   
    # Limit cache size to prevent memory issues
    if len(prediction_cache) > 10000:
        prediction_cache.clear()
   
    # Now execute all actions and update checkpoints
    for ai_idx in range(num_ais):
        if not is_alive[ai_idx]:
            continue
       
        action = actions[ai_idx]
        if action == 0:
            dot_positions[ai_idx][0] -= dot_speed
        elif action == 1:
            dot_positions[ai_idx][0] += dot_speed
        elif action == 2:
            dot_positions[ai_idx][1] -= dot_speed
        elif action == 3:
            dot_positions[ai_idx][1] += dot_speed




        x, y = dot_positions[ai_idx]
        onTrack = is_on_track(x, y)
       
        # Handle off-track counting (allow up to max_off_track consecutive off-track moves)
        if not onTrack:
            off_track_count[ai_idx] += 1
            if off_track_count[ai_idx] > max_off_track:
                # About to die - apply penalties
                is_alive[ai_idx] = False
                # Don't continue yet - we need to save the final state with penalties
        else:
            # Reset counter when back on track
            off_track_count[ai_idx] = 0
       
        # --- NEW CHECKPOINT SYSTEM ---
        # Check for checkpoint hits (in order)
        current_checkpoint = checkpoints_reached[ai_idx] % num_checkpoints
        cp_x, cp_y = checkpoints[current_checkpoint]
        dist_to_checkpoint = math.hypot(x - cp_x, y - cp_y)
       
        # Hit checkpoint if within radius and haven't hit it yet this lap
        if dist_to_checkpoint <= checkpoint_radius and not checkpoint_flags[ai_idx, current_checkpoint]:
            checkpoint_flags[ai_idx, current_checkpoint] = True
            checkpoints_reached[ai_idx] += 1
           
            # If we hit checkpoint 0 after completing all others, increment lap
            if current_checkpoint == 0 and checkpoints_reached[ai_idx] > num_checkpoints:
                laps[ai_idx] += 1
                max_laps[ai_idx] = max(max_laps[ai_idx], laps[ai_idx])
                # Reset checkpoint flags for new lap
                checkpoint_flags[ai_idx, :] = False
                checkpoint_flags[ai_idx, 0] = True  # Mark checkpoint 0 as hit for new lap
       
        # Calculate progress
        # Which checkpoint are we going toward next?
        next_checkpoint_idx = checkpoints_reached[ai_idx] % num_checkpoints
        next_cp_x, next_cp_y = checkpoints[next_checkpoint_idx]
       
        # Distance to next checkpoint
        dist_to_next = math.hypot(x - next_cp_x, y - next_cp_y)
       
        # Max expected distance (tune based on track)
        MAX_CP_DIST = 250.0
       
        # Progress toward next checkpoint (0 to 1)
        progress_to_next = max(0.0, min(1.0, 1.0 - dist_to_next / MAX_CP_DIST))
       
        # Total progress calculation
        # Each checkpoint passed = 20 points
        # Progress to next = up to 20 points
        # Lap bonus = 100 * lap number
        checkpoints_passed_this_lap = checkpoints_reached[ai_idx] % num_checkpoints
        total_progress = (checkpoints_passed_this_lap * 20.0) + (progress_to_next * 20.0) + (laps[ai_idx] * 100.0)
       
        # Use total_progress as base food
        food = total_progress
       
        # NEW PENALTIES
        # Off-track penalty: -10 food
        if not onTrack:
            food -= 100
       
        # About to die penalty: -50 food (applied when off_track_count exceeds max)
        if not is_alive[ai_idx]:
            food -= 50
            if(current_frame<150):
                food-=500
            # Check if dying near parent death position: additional -20 food
            for px, py in parent_death_positions.values():
                if math.hypot(x - px, y - py) <= 10:
                    food -= 10
                    break




        # Save
        food_array[ai_idx] = food
       
        # Save memory (state + action) - use total_progress as state info
        update_memory(ai_idx, x, y, onTrack, total_progress, laps[ai_idx], food, action)




    # --- Drawing ---
    screen.fill((100, 100, 100))
    track_color = (50, 50, 225)
   
    # Draw finish line (checkered)
    pygame.draw.rect(screen, (255,255,255), (380,365,20,60))
    pygame.draw.rect(screen, (0,0,0), (380,365,10,10))
    pygame.draw.rect(screen, (0,0,0), (390,375,10,10))
    pygame.draw.rect(screen, (0,0,0), (380,385,10,10))
    pygame.draw.rect(screen, (0,0,0), (390,395,10,10))
    pygame.draw.rect(screen, (0,0,0), (380,405,10,10))
    pygame.draw.rect(screen, (0,0,0), (390,415,10,10))
   
    pygame.draw.lines(screen, track_color, True, outer_boundary, 5)
    pygame.draw.lines(screen, track_color, True, inner_boundary, 5)
   
    # Draw checkpoints
    for i, cp in enumerate(checkpoints):
        pygame.draw.circle(screen, (255, 255, 0), (int(cp[0]), int(cp[1])), checkpoint_radius, 2)
        # Label checkpoints
        cp_label = font.render(str(i), True, (255, 255, 0))
        screen.blit(cp_label, (int(cp[0]) - 5, int(cp[1]) - 10))
   
    # Draw parent death positions (red X marks)
    for px, py in parent_death_positions.values():
        pygame.draw.line(screen, (255, 0, 0), (px - 10, py - 10), (px + 10, py + 10), 3)
        pygame.draw.line(screen, (255, 0, 0), (px - 10, py + 10), (px + 10, py - 10), 3)
        pygame.draw.circle(screen, (255, 0, 0), (int(px), int(py)), 25, 2)




    top_two = np.argsort(food_array)[-2:]
    dark_green = (0, 150, 0)




    for ai_idx in range(num_ais):
        x, y = dot_positions[ai_idx]




        if not is_alive[ai_idx]:
            color = (100, 100, 100)   # dead
        else:
            if ai_idx in top_two:
                color = dark_green    # highlight best two
            else:
                color = (0,255,0) if is_on_track(x, y) else (255,0,0)




        pygame.draw.circle(screen, color, (int(x), int(y)), 5)




    gen_text = font.render(f"Gen {generation} | Frame {current_frame}/{generation_length} | Eps {epsilon:.3f}", True, (255,255,255))
    screen.blit(gen_text, (10, 10))
   
    # Show alive count and best food
    alive_count = np.sum(is_alive)
    best_food = np.max(food_array)
   
    # Get lap counts and checkpoint progress of top two
    top_laps = [laps[top_two[0]], laps[top_two[1]]]
    top_checkpoints = [checkpoints_reached[top_two[0]] % num_checkpoints, checkpoints_reached[top_two[1]] % num_checkpoints]




    stats_text = font.render(
        f"Alive: {alive_count}/{num_ais} | Best: {best_food:.1f} | Top: L{top_laps[0]}C{top_checkpoints[0]}, L{top_laps[1]}C{top_checkpoints[1]}",
        True, (255,255,255)
    )
    screen.blit(stats_text, (10, 35))




    # --- Generation management ---
    current_frame += 1
   
    # Check if all AIs are dead or frame limit reached
    all_dead = not np.any(is_alive)
    if current_frame >= generation_length or all_dead:
        gen_start_time = time.time()  # Track how long generation transition takes
        if all_dead:
            print(f"\n=== Generation {generation} Complete (All AIs Dead at frame {current_frame}) ===")
            # Append best score of this generation
            best_score_this_gen = food_array.max()  # best among all agents this generation
            gen_history.append(generation)
            score_history.append(best_score_this_gen)


            # Update plot
            scatter_plot.set_offsets(np.c_[gen_history, score_history])
            line_plot.set_data(gen_history, score_history)


            ax.relim()
            ax.autoscale_view()


            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            print(f"\n=== Generation {generation} Complete (Frame Limit) ===")
       
        # Select top 2 performers first (needed for replay buffer and breeding)
        top_two = np.argsort(food_array)[-2:]
        print(f"Top performers: {top_two}, Food: {food_array[top_two]}, Laps: {laps[top_two]}, Checkpoints: {checkpoints_reached[top_two]}")
       
        # STORE PARENT DEATH POSITIONS BEFORE RESETTING
        parent_death_positions = {
            top_two[0]: tuple(dot_positions[top_two[0]]),
            top_two[1]: tuple(dot_positions[top_two[1]])
        }




        # 1) Add best experiences to global replay buffer
        for top_idx in top_two:
            mem = ai_memory[top_idx]
            if mem.shape[0] > 0:
                # Only add experiences with positive food
                good_experiences = mem[mem[:, 5] > 0]
                if len(good_experiences) > 0:
                    global_replay_buffer = np.concatenate((global_replay_buffer, good_experiences), axis=0)
       
        # Keep buffer size manageable
        if global_replay_buffer.shape[0] > replay_buffer_size:
            # Keep most recent experiences
            global_replay_buffer = global_replay_buffer[-replay_buffer_size:]
       
        print(f"Replay buffer size: {global_replay_buffer.shape[0]}")
       
        # 2) CONDITIONAL TRAINING: Only train every N generations to save time
        if generation % train_every_n_generations == 0:
            train_start = time.time()
            print(f"Training agents...")
           
            # Pre-build replay buffer training data ONCE (reuse for all agents)
            X_replay, y_replay = build_training_data(global_replay_buffer, seq_len=seq_len, min_food_threshold=10.0)
            if X_replay.shape[0] > 0:
                y_replay_cat = tf.keras.utils.to_categorical(y_replay, num_classes=4)
           
            trained_count = 0
            for ai_idx in range(num_ais):
                # Train on own good experiences
                mem = ai_memory[ai_idx]
                X_own, y_own = build_training_data(mem, seq_len=seq_len, min_food_threshold=10.0)
               
                # Combine datasets
                if X_own.shape[0] > 0 and X_replay.shape[0] > 0:
                    X_train = np.concatenate((X_own, X_replay), axis=0)
                    y_train = np.concatenate((y_own, y_replay), axis=0)
                    y_cat = tf.keras.utils.to_categorical(y_train, num_classes=4)
                elif X_replay.shape[0] > 0:
                    X_train = X_replay
                    y_cat = y_replay_cat
                elif X_own.shape[0] > 0:
                    X_train = X_own
                    y_train = y_own
                    y_cat = tf.keras.utils.to_categorical(y_train, num_classes=4)
                else:
                    continue
               
                # Limit training samples to max 1000 to prevent slowdown
                if X_train.shape[0] > 1000:
                    indices = np.random.choice(X_train.shape[0], 1000, replace=False)
                    X_train = X_train[indices]
                    y_cat = y_cat[indices]
               
                # Train with reduced verbosity
                try:
                    models[ai_idx].fit(X_train, y_cat, epochs=train_epochs_per_agent,
                                     batch_size=batch_size, verbose=0)
                    trained_count += 1
                except Exception as e:
                    print(f"  Train error agent {ai_idx}: {e}")
           
            train_time = time.time() - train_start
            print(f"  Trained {trained_count} agents in {train_time:.2f}s")
        else:
            print(f"Skipping training this generation (train every {train_every_n_generations})")
       
        # 3) Transfer weights to other agents (FAST - no model creation!)
        breed_start = time.time()
        parent_weights = [models[idx].get_weights() for idx in top_two]
       
        # Assign children (reuse existing model objects)
        # NEW: Track which children belong to which parent for epsilon assignment
        child_parent_map = []
        child_counts = [0, 0]  # Track how many children per parent
       
        child_idx = 0
        for parent_idx in range(2):
            for _ in range(num_ais // 2):
                if child_idx not in top_two:  # Don't overwrite parents
                    transfer_weights_mutated(models[child_idx], parent_weights[parent_idx], std=mutation_std)
                    child_parent_map.append(top_two[parent_idx])
                    child_counts[parent_idx] += 1
                else:
                    child_parent_map.append(child_idx)  # Parent maps to itself




                child_idx += 1
       
        # Handle any remaining slots
        while child_idx < num_ais:
            if child_idx not in top_two:
                transfer_weights_mutated(models[child_idx], parent_weights[0], std=mutation_std)
                child_parent_map.append(top_two[0])
                child_counts[0] += 1
            else:
                child_parent_map.append(child_idx)  # Parent maps to itself
            child_idx += 1
       
        breed_time = time.time() - breed_start
        print(f"Breeding took {breed_time:.3f}s")




        # 4) Reset positions/memory/food for next generation
        new_positions = []
        for parent_idx in top_two:
            for _ in range(num_ais // 2):
                px, py = dot_positions[parent_idx]
                nx = START_X + random.uniform(-10, 10) + (px - START_X) * 0.1
                ny = START_Y + random.uniform(-10, 10) + (py - START_Y) * 0.1
                new_positions.append([nx, ny])
        new_positions = new_positions[:num_ais]
        dot_positions = np.array(new_positions, dtype=float)




        # Reset per-agent stats
        food_array[:] = 0.0
        laps[:] = 0
        max_laps[:] = 0
        checkpoints_reached[:] = 0
        checkpoint_flags[:, :] = False
        finished_last_frame[:] = False
        is_alive[:] = True
        off_track_count[:] = 0
        ai_memory = [np.zeros((0, 7), dtype=float) for _ in range(num_ais)]




        # Reset agent epsilons to global epsilon
        agent_epsilons[:] = epsilon
       
        # NEW: Set high exploration (90%) for 2 children per parent within 25 pixels of death
        high_epsilon_count = [0, 0]  # Track per parent
        for child_idx in range(num_ais):
            if child_idx in top_two:
                continue  # Don't modify parents
           
            parent_idx = child_parent_map[child_idx]
            parent_idx_in_top = 0 if parent_idx == top_two[0] else 1
           
            # Only set high epsilon for first 2 children per parent
            if high_epsilon_count[parent_idx_in_top] >= 2:
                continue
           
            parent_death_x, parent_death_y = parent_death_positions[parent_idx]
            child_x, child_y = dot_positions[child_idx]
           
            # Check if within 15 pixels of where this parent died
            dist = math.hypot(child_x - parent_death_x, child_y - parent_death_y)
           
            if dist <= 35:
                agent_epsilons[child_idx] = 0.89  # 90% exploration
                high_epsilon_count[parent_idx_in_top] += 1
                print(f"  Child {child_idx} (parent {parent_idx}) within {dist:.1f}px of death - high exploration")




        # decay epsilon (less random exploration next generation)
        if epsilon == 0.9:
            epsilon = 0.5
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
       
        gen_total_time = time.time() - gen_start_time
        print(f"New epsilon: {epsilon:.4f}")
        print(f"Total generation transition: {gen_total_time:.2f}s\n")


        current_frame = 0
        generation += 1



    pygame.display.flip()
    clock.tick(800)


pygame.quit()


