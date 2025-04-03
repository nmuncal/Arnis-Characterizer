import cv2
from ultralytics import YOLO
import argparse
import os
import csv
import torch
import datetime
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import torch.nn as nn
from ultralytics.utils.plotting import Annotator
from collections import Counter




# Arguments
parser = argparse.ArgumentParser(description='YOLOv8-pose')
parser.add_argument('--source', type=str, default='webcam', choices=['webcam', 'video'],
                    help='Source for pose estimation: webcam or video')
parser.add_argument('--action', type=str, default='extract', choices=['extract', 'action_recognition'],
                    help='Type of action: extraction for training or action recognition')
parser.add_argument('--all', type=bool, default=False,
                    help='Check if all videos or a single video will be processed')
parser.add_argument('--video_path', type=str, help='Path to the video file if source is video')
parser.add_argument('--pattern_checking', type=bool,default=True)
parser.add_argument('--resolution', type=str, default='1080x1920', help='Input resolution for YOLOv8 (example usage: 640x640)')
parser.add_argument('--min_sequence_length', type=int, default=20, help='Minimum length for the sequence')
parser.add_argument('--pattern_length', type=int, default=3, help='Length of output pattern')



args = parser.parse_args()

# Load YOLOv8 pose model
model = YOLO("yolov8l-pose.pt")

# Move model to GPU if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load LSTM model and Label Encoder
if args.action == 'action_recognition':
    lstm_model = load_model('front_model2.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

# Parse resolution
width, height = map(int, args.resolution.split('x'))

# For normalizing keypoints to range [0, 1]
def normalize_keypoints(keypoints, width, height, action):
    keypoints_np = keypoints.cpu().numpy() if keypoints.is_cuda else keypoints.numpy()

    # Normalize the keypoints
    normalized = keypoints_np.copy()
    normalized[:, 0] /= width  # Normalize x-coordinates by image width
    normalized[:, 1] /= height  # Normalize y-coordinates by image height

    if action == 'action_recognition':
        return normalized.flatten()
    return normalized.tolist()


def save_moves_to_file(person_states, filename="person_states.txt"):
    with open(filename, 'w') as file:
        for person_id, state_info in person_states.items():
            file.write(f"Person ID: {person_id}\n")
            file.write(f"Moves: {state_info['moves']}\n")
            counts = Counter(state_info['moves'])
            file.write("\n")
            for item, count in counts.items():
                file.write(f"{item}: {count} \n")
            file.write("\n")
            for i, (seq, count) in enumerate(state_info['sequence_counts']):
                pattern_text = " -> ".join(map(str, seq))
                file.write(f"{i+1}. {pattern_text} (Count: {count})\n")
    print(f"Moves saved to {filename}")


def calculate_angle(p1, p2, p3):
    p1 = p1.cpu().numpy() if isinstance(p1, torch.Tensor) else np.array(p1)
    p2 = p2.cpu().numpy() if isinstance(p2, torch.Tensor) else np.array(p2)
    p3 = p3.cpu().numpy() if isinstance(p3, torch.Tensor) else np.array(p3)

    vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])  
    vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])  

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return np.nan  # Undefined

    cos_theta = np.dot(vector1, vector2) / (norm1 * norm2)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)
    return np.degrees(angle)



def check_abierta(keypoints, person_id):

    right_knee, left_knee = keypoints[14], keypoints[13]
    right_ankle, left_ankle = keypoints[16], keypoints[15]
    right_shoulder, left_shoulder = keypoints[6], keypoints[5]
    right_elbow, left_elbow = keypoints[8], keypoints[7]
    right_wrist, left_wrist = keypoints[10], keypoints[9]
    right_ear, left_ear = keypoints[4], keypoints[7]

    # Calculate elbow angle
    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    print(f"Person ID: {person_id}")


    if right_ankle[0] < left_ankle[0] and right_ankle[1] > left_ankle[1]:
        print("true")
        if right_elbow[0] < right_shoulder[0] and right_elbow[1] > right_wrist[1] and right_shoulder[1] > right_wrist[1] and right_wrist[0] < right_shoulder[0]:
            print("true")
            if left_wrist[0] < left_elbow[0] and left_wrist[1] < left_elbow[1]:
                print(elbow_angle)
                if elbow_angle <= 80:
                    print("ABIERTA")
                    return True
            
            
    return False


def check_serrada(keypoints, person_id):
    right_knee, left_knee = keypoints[14], keypoints[13]
    right_ankle, left_ankle = keypoints[16], keypoints[15]
    right_shoulder, left_shoulder = keypoints[6], keypoints[5]
    right_elbow, left_elbow = keypoints[8], keypoints[7]
    right_wrist, left_wrist = keypoints[10], keypoints[9]
    right_hip, left_hip = keypoints[12],keypoints[11]
    right_ear, left_ear = keypoints[4], keypoints[7]

    # Calculate elbow angle
    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    print(f"Person ID: {person_id}")


    if right_ankle[0] < left_ankle[0] and right_ankle[1] > left_ankle[1] and right_wrist[0] > right_hip[0]:
        if right_shoulder[1] < right_elbow[1]:
            if right_elbow[0] < right_wrist[0] and right_elbow[0] > right_shoulder[0] and right_wrist[0] > left_shoulder[0] and right_wrist[1] > left_shoulder[1] :
                if left_wrist[2] >= 0.50 and left_elbow[2] >= 0.50:
                    if left_wrist[1] > left_elbow[1] and right_wrist[0] > left_elbow[0] and left_wrist[0] > left_elbow[0]:
                        print(elbow_angle)
                        if elbow_angle <= 120:
                            # abs(right_wrist[0] - right_elbow[0])
                            print(abs(right_wrist[0] - right_elbow[0]))
                            if abs(right_wrist[1] - right_elbow[1]) < 75 and abs(right_wrist[0] - right_elbow[0]) <= 90:
                                print("SERRADA")
                                return True
                else:

                    print(elbow_angle)
                    if elbow_angle <= 120:

                        print(abs(right_wrist[1] - right_elbow[1]))
                        if abs(right_wrist[1] - right_elbow[1]) < 75 and abs(right_wrist[0] - right_elbow[0]) <= 90:
                            print("SERRADA")
                            return True
            
    return False

def check_pattern(moves,length):
    patterns = []

    #Check if moves list is empty or pattern length is greater than the current number of moves in the list
    if not moves or length <= 0 or length > len(moves):
        return [] 

    #Extract Patterns
    for i in range(len(moves)- length +1):
        sequence = tuple(moves[i:i+length])
        patterns.append(sequence)

    #Return Top 5 most common
    return Counter(patterns).most_common(3)



def draw_text(frame, text, pos, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


def process_source(source, video_path=None):
    if source == 'webcam':
        cap = cv2.VideoCapture(1)  # 0 - webcam, 1- phone camera
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    elif source == 'video':
        if video_path is None:
            raise ValueError("Video path must be provided when source is 'video'")
        cap = cv2.VideoCapture(video_path)
    else:
        raise ValueError("Invalid source. Choose 'webcam' or 'video'")

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if source == 'video':
        title = f'{video_path}_{current_datetime}_output.mp4'
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        csv_filename = os.path.join(video_dir, f'{video_name}_keypoints_{current_datetime}.csv')
    else:
        title = f'{source}_{current_datetime}_output.mp4'
        csv_filename = f'{source}_keypoints_{current_datetime}.csv'

    out = cv2.VideoWriter(title, fourcc, fps, (width, height))

    if args.action == 'extract':
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
    
        # Updated header to reflect normalized coordinates
        header = ['video_title', 'action', 'person_id', 'frame']
        for i in range(17):
            header.extend([f'kp{i}_x', f'kp{i}_y'])
        csv_writer.writerow(header)

    frame_count = 0
    sequence_length = 90
    person_states = {}
    min_sequence_length = args.min_sequence_length
    all_person_ids = set()
    action_mapping = {
        "Downward Diagonal Strike - Abierta": 1,
        "Downward Diagonal Strike - Serrada": 2,
        "Horizontal Strike - Abierta": 3,
        "Horizontal Strike - Serrada": 4,
        "Horizontal Strike Head - Abierta": 5,
        "Horizontal Strike Head - Serrada": 6,
        "Stab Eye/Chest - Abierta": 7,
        "Stab Eye/Chest - Serrada": 8,
        "Stab Solar Plexus": 9,
        "Upward Diagonal Strike - Abierta": 10,
        "Upward Diagonal Strike - Serrada": 11,
        "Vertical Strike - Serrada": 12
    }

    while True:
        success, frame = cap.read()
        if not success:
            # Process any incomplete sequences before breaking
            if args.action == 'action_recognition':
                for person_id, state in person_states.items():
                    if state['state'] == 'COLLECTING_SEQUENCE' and len(state['sequence']) >= min_sequence_length:  # minimum sequence length
                        # Pad sequence if needed

                        zeros_array = [
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0
                        ]

                        if len(state['sequence']) < sequence_length:
                            padding = [zeros_array] * (sequence_length - len(state['sequence']))
                            full_sequence = state['sequence'] + padding
                        else:
                            full_sequence = state['sequence'][-sequence_length:]

                        data = np.reshape(full_sequence, (1, sequence_length, 34))

                        try:
                            lstm_output = lstm_model.predict(data)
                            action_index = np.argmax(lstm_output)
                            action_name = le.inverse_transform([action_index])[0]
                            confidence = lstm_output[0][action_index]

                            state['last_action'] = {
                                'name': action_name,
                                'confidence': confidence
                            }
                            state['moves'].append(action_name)
                            print(f"Final action for person {person_id}: {action_name} ({confidence:.2f})")
                        except Exception as e:
                            print(f"Error processing final sequence for person {person_id}: {e}")
            
            break
        
        frame_count += 1
        
        # Resize frame to the specified resolution if it doesn't match 640x640
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
        # Perform tracking
        results = model.track(frame, persist=True,tracker="bytetrack.yaml")
        annotator = Annotator(frame, line_width=2)

        for r in results:
            if r.keypoints is not None:
                keypoints = r.keypoints.data  # Get keypoints
                if r.boxes is not None and r.boxes.id is not None:
                    ids = r.boxes.id.int().cpu().tolist()  # Get ids as a list of integers
                    all_person_ids.update(ids)  
                    boxes = r.boxes.xywh.cpu()
                    clss = r.boxes.cls.cpu().tolist()

                    for person_id, person_keypoints, box, cls in zip(ids, keypoints, boxes, clss):
                        # Normalize keypoints
                        normalized_keypoints = normalize_keypoints(person_keypoints[:, :2], width, height, args.action)
                        x, y, w, h = box
                        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2

                        annotator.kpts(person_keypoints, shape=frame.shape)
                        if args.action == "extract":
                            annotator.box_label([x1, y1, x2, y2], f"person id: {person_id}", color=(0, 0, 255))

                        if args.action == 'extract':
                            # Prepare the row data
                            row_data = [title, 'unknown', person_id, frame_count]
                            for kp in normalized_keypoints:
                                row_data.extend(kp[:2])  # This adds x, y, conf for each keypoint
                            csv_writer.writerow(row_data)


                        if args.action == 'action_recognition':
                            # Initialize state for this person if not exists
                            if person_id not in person_states:
                                person_states[person_id] = {
                                    'state': 'WAITING_STARTING_POSE',  
                                    'sequence': [],
                                    'moves' : [],
                                    'moves_integer':[],
                                    'sequence_counts':[],
                                    'last_action': None
                                }

                            # Check for starting poses
                            
                            abierta = check_abierta(person_keypoints, person_id)

                            if not abierta:
                                serrada = check_serrada(person_keypoints, person_id)

                            current_state = person_states[person_id]['state']

                            if args.pattern_checking:
                                # Get the top 5 action sequences
                                sequence_counts = check_pattern(person_states[person_id]['moves_integer'], args.pattern_length)


                        
                                person_states[person_id]['sequence_counts'] = sequence_counts
                                print("Top Sequences:")
                                for i in range(3):
                                    if i < len(sequence_counts):
                                        seq, count = sequence_counts[i]
                                        pattern_text = f"{i+1}. {' -> '.join(map(str, seq))}"
                                    else:
                                        pattern_text = f"{i+1}. ---"
                                    
                                    print(pattern_text)


                                

                            else:
                                print("Pattern checking disabled")     

                            if current_state == 'WAITING_STARTING_POSE':
                                # Look for starting pose
                                if abierta:
                                    # Detected starting pose, prepare to collect sequence
                                    person_states[person_id]['state'] = 'COLLECTING_SEQUENCE'
                                    person_states[person_id]['sequence'] = []
                                    pose_text = "Abierta"
                                    if len(ids) == 1:
                                        draw_text(frame, f"Starting Pose Detected: {pose_text}", (20, 40), (0, 0, 255))
                                    else:
                                        print(f"Starting Pose Detected: {pose_text}")
    
                                elif 'serrada' in locals() and serrada:  
                                    # Detected starting pose, prepare to collect sequence
                                    person_states[person_id]['state'] = 'COLLECTING_SEQUENCE'
                                    person_states[person_id]['sequence'] = []
                                    pose_text = "Serrada"
                                    if len(ids) == 1:
                                        draw_text(frame, f"Starting Pose Detected: {pose_text}", (20, 40), (0, 0, 255))
                                    else:
                                        print(f"Starting Pose Detected: {pose_text}")
                                        

                            elif current_state == 'COLLECTING_SEQUENCE':

                                min_sequence_length = args.min_sequence_length
                                target_sequence_length = sequence_length  # 90

                                if len(person_states[person_id]['sequence']) >= min_sequence_length:
                                    if abierta or serrada or  len(person_states[person_id]['sequence']) >= target_sequence_length:
                                        if len(person_states[person_id]['sequence']) < target_sequence_length:
                                            zeros_array = [
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                                0.0, 0.0, 0.0, 0.0
                                            ]
                                            padding = [zeros_array] * (target_sequence_length - len(person_states[person_id]['sequence']))
                                            full_sequence = person_states[person_id]['sequence'] + padding
                                        else:
                                            full_sequence = person_states[person_id]['sequence'][-target_sequence_length:]
                                                
                                        data = np.reshape(full_sequence, (1, target_sequence_length, 34))

                                        try:
                                            lstm_output = lstm_model.predict(data)
                                            action_index = np.argmax(lstm_output)
                                            action_name = le.inverse_transform([action_index])[0]
                                            confidence = lstm_output[0][action_index]

                                            person_states[person_id]['last_action'] = {
                                                'name': action_name,
                                                'confidence': confidence
                                            }

                                            person_states[person_id]['moves'].append(action_name)

                                            if action_name in action_mapping:
                                                action_int = action_mapping[action_name]  # Get corresponding integer
                                                person_states[person_id]['moves_integer'].append(action_int)

                                            print(person_states[person_id]['moves'])
                                            

                                            # annotator.box_label([x1, y1, x2, y2], 
                                            #                     f"Action: {action_name}, Conf: {confidence:.2f}", 
                                            #                     color=(0, 0, 255))

                                            # draw_text(frame, f"Detected Action: {action_name} ({confidence:.2f})", (20, 70), (255, 0, 0))

                                        except Exception as e:
                                            print(f"Prediction error for person {person_id}: {e}")

                                        person_states[person_id]['state'] = 'WAITING_STARTING_POSE'
                                        person_states[person_id]['sequence'] = []
                                    else:
                                        person_states[person_id]['sequence'].append(normalized_keypoints)
                                else:

                                    if abierta or serrada:

                                        person_states[person_id]['sequence'] = []
                                        if len(ids) == 1:
                                            if abierta:
                                                draw_text(frame, "Starting Pose Detected, Resetting Sequence -> Abierta", (20, 130), (255,0,0))
                                            else:
                                                draw_text(frame, "Starting Pose Detected, Resetting Sequence -> Serrada", (20, 130), (255,0,0))
                                        else:
                                            print("Starting Pose Detected, Resetting Sequence")

                                    else:
                                        person_states[person_id]['sequence'].append(normalized_keypoints)

                            if person_states[person_id]['last_action']:
                                last_action = person_states[person_id]['last_action']
                                if len(ids) > 1:
                                    annotator.box_label([x1, y1, x2, y2], f"Last Action: {last_action['name']}, Conf: {last_action['confidence']:.2f}", 
                        color=(0, 0, 255))
                                if len(ids) == 1:
                                    draw_text(frame, f"Last Action: {last_action['name']} ({last_action['confidence']:.2f})", (20, 160),(0, 255, 255))
                                else:
                                    print(f"Last Action: {last_action['name']} ({last_action['confidence']:.2f})")

                            state_text = {
                                'WAITING_STARTING_POSE': 'Waiting for Starting Pose',
                                'COLLECTING_SEQUENCE': 'Collecting Sequence'
                            }[person_states[person_id]['state']]

                            sequence_length_text = f"Sequence Length: {len(person_states[person_id]['sequence'])}"

                            print(sequence_length_text)
                            if len(ids) == 1:
                                draw_text(frame, sequence_length_text, (20, 190), (0, 165, 255))
                                print(sequence_length_text)
                            if len(ids) == 1:
                                draw_text(frame, f"State: {state_text}", (20, 70), (255,0,0))
                            else:
                                print(f"State: {state_text}")



        # Plot results
        annotated_frame = annotator.result()

        window_name = f'{source.capitalize()} Feed'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        display_width = 1080
        display_height = 1920
        cv2.resizeWindow(window_name, display_width, display_height)
        cv2.imshow(window_name, annotated_frame)
        out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release resources
    cap.release()

    if args.action == 'extract':
        if frame_count < sequence_length:
            padding_frames = sequence_length - frame_count
            print(f"Adding {padding_frames} frames of padding using last known keypoints")
            
            # Determine the number of keypoints (based on the normalized keypoints structure)
            num_keypoints = len(normalized_keypoints) if 'normalized_keypoints' in locals() else 0

            for pad_frame in range(frame_count + 1, sequence_length + 1):  
                for person_id in all_person_ids:  
                    row_data = [title, 'unknown', person_id, pad_frame]
                    # Add zeroes for each keypoint (x, y)
                    for _ in range(num_keypoints):
                        row_data.extend([0.0, 0.0])  # Zeroes for x, y
                    csv_writer.writerow(row_data)
            
            # Close the CSV file after writing all padding frames
            csv_file.close()
            print(f"Normalized keypoints saved to {csv_filename}")
    else:

        if args.source == 'webcam':
            text_file_name = f'webcam_moves_{current_datetime}.txt'
        else:
            text_file_name = f'{video_name}_moves_{current_datetime}.txt'


        save_moves_to_file(person_states,text_file_name)


    cv2.destroyAllWindows()

if __name__ == "__main__":
    if args.source == 'video' and args.video_path is None:
        parser.error("--video_path is required when --source is 'video'")

    if args.all:

        for subfolder in os.listdir(args.video_path):
            subfolder_path = os.path.join(args.video_path, subfolder)

            if os.path.isdir(subfolder_path):
                for video_file in os.listdir(subfolder_path):
                    if video_file.endswith(".mp4"):
                        print(f"Processing video: {video_file}")
                        process_source(args.source, os.path.join(subfolder_path, video_file))

                # # Process all videos in the folder
                # print("Processing all videos")
                # for filename in os.listdir(args.video_path):
                #     if filename.lower().endswith(('.mp4', '.mov')):
                #         print(f"Processing video: {filename}")
                #         process_source(args.source, os.path.join(args.video_path, filename))


    else:
        process_source(args.source, args.video_path)