import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Configuration Parameters
# ================================

# Video path
video_path = '' # the location of the video goes here

# YOLO model path
model_path = 'best.pt'

# Tracker configuration
tracker_config = 'bytetrack.yaml'  # Ensure this YAML file is correctly configured

# Output text file
output_file = 'withoutmodifications.txt'

# User inputs
pixel_to_micron_ratio = float(input("Enter pixel to micron ratio: "))  # Conversion factor from pixel to microns
chamber_height = float(input("Enter chamber height in microns: "))  # Height of the chamber
original_frame_rate = float(input("Enter the original frame rate: "))  # Original frame rate of the video
maximum_threshold = float(input("Enter the maximum threshold in dimensions (in microns): "))

# Stabilization parameters
history_length = 5          # Number of frames to consider for stabilization
size_change_threshold = pixel_to_micron_ratio * maximum_threshold   # Maximum allowed change in width or height (pixels) between consecutive frames

# ================================
# Initialize Video Capture
# ================================

# Load the video
video = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not video.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

# Get the frame rate of the video
frame_rate = video.get(cv2.CAP_PROP_FPS)
print(f"Frame Rate: {frame_rate} FPS")

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
output_video_path = 'output_with_bounding_boxes.mp4'

# Initialize VideoWriter
out_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate/10 , (frame_width, frame_height))

# Calculate delay per frame in milliseconds
if frame_rate > 0:
    delay = int(1000 / frame_rate)
else:
    delay = 1  # Default delay

# ================================
# Load YOLO Model
# ================================

# Initialize YOLO model
model = YOLO(model_path)

# ================================
# Initialize Data Structures
# ================================

recorded_volumes = []  # List to store volumes of all droplets
total_frames = 0       # Total number of frames processed
recorded_ids = set()   # Track objects already processed

# Dictionary to store history for stabilization
# Format: {obj_id: {'width_history': [...], 'height_history': [...], 'stabilized': False}}
object_histories = {}
frame_volume_data = []
droplet_count_data = []

# ================================
# Open Output File
# ================================

with open(output_file, 'w') as file:
    # Write header
    file.write("Droplet Volumes (picolitres)\n")

    # ================================
    # Process Each Frame
    # ================================

    results = model.track(source=video_path, show=True, tracker=tracker_config)

    for result in results:
        total_frames += 1  # Increment frame count
        frame = result.orig_img  # Original frame for visualization
        stabilised_volumes = []

        for obj in result.boxes:
            obj_id = int(obj.id)  # Object ID

            # Extract object information
            bbox = obj.xyxy.tolist()[0]  # Bounding box coordinates [x1, y1, x2, y2]
            width = bbox[2] - bbox[0]    # Width of the bounding box (diameter of droplet in pixels)
            height = bbox[3] - bbox[1]   # Height of the bounding box (length of droplet in pixels)

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0), 2) # green bounding box
            label = f"ID: {obj_id}, Conf: {float(obj.conf)}"
            cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0), 2)

            # Initialize history for new objects
            if obj_id not in object_histories:
                object_histories[obj_id] = {
                    'width_history': [],
                    'height_history': [],
                    'stabilized': False
                }

            # Append current width and height to history
            object_histories[obj_id]['width_history'].append(width)
            object_histories[obj_id]['height_history'].append(height)

            # Maintain history length
            if len(object_histories[obj_id]['width_history']) > history_length:
                object_histories[obj_id]['width_history'].pop(0)
                object_histories[obj_id]['height_history'].pop(0)

            # Check if the object is stabilized
            if not object_histories[obj_id]['stabilized']:
                width_history = object_histories[obj_id]['width_history']
                height_history = object_histories[obj_id]['height_history']

                if len(width_history) == history_length:
                    # Calculate max change in width and height
                    width_changes = [abs(width_history[i] - width_history[i - 1]) for i in range(1, history_length)]
                    height_changes = [abs(height_history[i] - height_history[i - 1]) for i in range(1, history_length)]

                    max_width_change = max(width_changes)
                    max_height_change = max(height_changes)

                    if max_width_change <= size_change_threshold and max_height_change <= size_change_threshold:
                        # Stabilized, calculate volume
                        mean_width = np.mean(width_history) * pixel_to_micron_ratio  # Convert to microns
                        mean_height = np.mean(height_history) * pixel_to_micron_ratio  # Convert to microns

                        if mean_width <= chamber_height:
                            # Sphere droplet
                            radius = mean_width / 2
                            h = chamber_height
                            volume = (4 * np.pi * radius ** 3) / (3 * 1000)  # Volume in picolitres
                            print(f"id={obj_id}, sphere,diameter = {mean_width}, height = {h}, volume = {volume} pL\n")

                        elif mean_width > chamber_height and abs(mean_height - mean_width) < 0.175*(mean_width+mean_height)/2:
                            # Disk droplet
                            radius = (mean_width+mean_height) / 4
                            h = chamber_height
                            volume = ((2 * np.pi * (h / 2) * (radius - (h / 2)) ** 2) +
                                      (np.pi ** 2 * (h / 2) ** 2 * (radius - (h / 2)) * (1 - 4 / (3 * np.pi)))) / 1000
                            print(f"id={obj_id}, disc,diameter = {2*radius}, height = {h}, volume = {volume} pL\n")

                        else:
                            # Plug droplet (from the formula in the image)
                            H = chamber_height
                            L = mean_height
                            W = mean_width
                            c = 0.5
                            volume = (( H * W - (4 - np.pi)*((2/H) + (2/W))** -2 - c * H ** 2)*(L - W / 3)) / 1000
                            print(f"id={obj_id}, plug, length = {W}, width= {L}, height = {H}, volume = {volume} pL\n")
                        # Write the volume to file and store it in the list
                        file.write(f"{volume:.2f}\n")
                        recorded_volumes.append(volume)
                        stabilised_volumes.append(volume)

                        # Mark object as stabilized
                        object_histories[obj_id]['stabilized'] = True
                        recorded_ids.add(obj_id)

        if stabilised_volumes:
            real_time = total_frames / original_frame_rate
            mean_volume = np.mean(stabilised_volumes)
            frame_volume_data.append((real_time,mean_volume))
            droplet_count_data.append((real_time,(len(recorded_ids)/total_frames)))

        out_video.write(frame)
    # ================================
    # After Processing All Frames
    # ================================

    # Calculate generation speed
    generating_speed_per_frame = len(recorded_ids) / total_frames
    generating_speed_per_second = generating_speed_per_frame * original_frame_rate 

    # Write generating speed and average volume
    avg_volume = np.mean(recorded_volumes) if recorded_volumes else 0
    file.write(f"\nAverage Volume: {avg_volume:.2f} picolitres\n")
    file.write(f"Generation Speed: {generating_speed_per_second:.2f} droplets per second\n")

    # Print generating speed and average volume
    print(f"Average Volume: {avg_volume:.2f} picolitres")
    print(f"Generation Speed: {generating_speed_per_second:.2f} droplets per second")

time_vol, volumes = zip(*frame_volume_data)
time_gen, droplet_counts = zip(*droplet_count_data)



# Plot Volume vs Frames
plt.figure(figsize=(10, 5))
plt.plot(time_vol, volumes, label='Mean Volume (picolitres)', color='b', marker='o')
plt.xlabel('Time (in seconds)')
plt.ylabel('Mean Volume (picolitres)')
plt.title('Volume vs Time')
plt.ylim(0, max(volumes)*2)  # Adjust Y-axis range
plt.legend()
plt.grid()
plt.savefig('volume_vs_frames.png')  # Save the plot
plt.show()

# Plot Droplets vs Frames
plt.figure(figsize=(10, 5))
plt.plot(time_gen, droplet_counts, label='Generation Speed (droplets per second)', color='r', marker='s')
plt.xlabel('Time (in seconds)')
plt.ylabel('Generation Speed (droplets per second)')
plt.title('Generation Speed vs Seconds')
plt.ylim(0, max(droplet_counts)*2)  # Adjust Y-axis range
plt.legend()
plt.grid()
plt.savefig('droplets_vs_frames.png')  # Save the plot
plt.show()

# ================================
# Cleanup
# ================================

video.release()
out_video.release()
cv2.destroyAllWindows()
