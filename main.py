import cv2
import mediapipe as mp
import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk

# Initialize Mediapipe Pose globally
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class ExerciseDetectionApp:
    def _init_(self, root):
        self.root = root
        self.root.title("PERSONAL AI GYM")

        # Title label with bold and stylish font
        title_label = tk.Label(root, text="PERSONAL AI GYM", font=('Helvetica', 20, 'bold'), pady=20)
        title_label.pack()

        # Create buttons with more color and increased size
        self.squat_button = tk.Button(root, text="Squat Detection", command=self.start_squat_detection,
                                      bg='#4CAF50', fg='white', font=('Helvetica', 14), padx=20, pady=10)
        self.squat_button.pack(pady=10)

        self.pushup_button = tk.Button(root, text="Push-up Detection", command=self.start_pushup_detection,
                                       bg='#2196F3', fg='white', font=('Helvetica', 14), padx=20, pady=10)
        self.pushup_button.pack(pady=10)

        self.side_plank_button = tk.Button(root, text="Side Plank Detection", command=self.start_side_plank_detection,
                                           bg='#FF8C00', fg='white', font=('Helvetica', 14), padx=20, pady=10)
        self.side_plank_button.pack(pady=10)

        self.bird_dog_button = tk.Button(root, text="Bird-Dog Detection", command=self.start_bird_dog_detection_thread,
                                         bg='#FFD700', fg='white', font=('Helvetica', 14), padx=20, pady=10)
        self.bird_dog_button.pack(pady=10)

        # Initialize variables for exercise detection
        self.init_exercise_variables()

        # Set up video capture
        self.cap = None

    def init_exercise_variables(self):
        # Initialize variables for all exercises
        self.exercise_vars = {
            'squat': {'counter': 0, 'in_position': False, 'start_time': 0, 'end_time': 0, 'weight_kg': 54, 'met_value': 5.0},
            'pushup': {'counter': 0, 'in_position': False, 'start_time': 0, 'end_time': 0, 'weight_kg': 54, 'met_value': 8.0},
            'side_plank': {'counter': 0, 'in_position': False, 'start_time': 0, 'end_time': 0, 'weight_kg': 70, 'met_value': 2.9},
            'bird_dog': {'counter': 0, 'in_position': False, 'start_time': 0, 'end_time': 0, 'weight_kg': 54, 'met_value': 2.6}
        }

    def start_exercise_detection(self, exercise_type):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                self.process_landmarks(results.pose_landmarks.landmark, exercise_type, frame)

            # Display the frame
            cv2.imshow('Exercise Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_landmarks(self, landmarks, exercise_type, frame):
        # Extract landmarks and perform exercise-specific detection
        if exercise_type == 'squat':
            self.detect_squat(landmarks, frame)
        elif exercise_type == 'pushup':
            self.detect_pushup(landmarks, frame)
        elif exercise_type == 'side_plank':
            self.detect_side_plank(landmarks, frame)
        elif exercise_type == 'bird_dog':
            self.detect_bird_dog(landmarks, frame)

        # Draw landmarks on the frame
        self.draw_landmarks(landmarks, frame)

        # Display exercise counter and calories
        self.display_exercise_info(frame, exercise_type)

    def detect_squat(self, landmarks, frame):
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

        if hip_left < knee_left and hip_right < knee_right:
            self.update_exercise_state('squat', True)
        else:
            self.update_exercise_state('squat', False)

    def detect_pushup(self, landmarks, frame):
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        elbow_right = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y

        if shoulder_left > elbow_left and shoulder_right > elbow_right:
            self.update_exercise_state('pushup', True)
        else:
            self.update_exercise_state('pushup', False)

    def detect_side_plank(self, landmarks, frame):
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ankle_left = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y

        if shoulder_left < hip_left < ankle_left:
            self.update_exercise_state('side_plank', True)
        else:
            self.update_exercise_state('side_plank', False)

    def detect_bird_dog(self, landmarks, frame):
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ankle_right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

        if shoulder_left < hip_left and knee_right < ankle_right:
            self.update_exercise_state('bird_dog', True)
        else:
            self.update_exercise_state('bird_dog', False)

    def update_exercise_state(self, exercise_type, in_position):
        vars = self.exercise_vars[exercise_type]
        if in_position and not vars['in_position']:
            vars['start_time'] = cv2.getTickCount()
            vars['in_position'] = True
        elif not in_position and vars['in_position']:
            vars['end_time'] = cv2.getTickCount()
            duration = (vars['end_time'] - vars['start_time']) / cv2.getTickFrequency()
            if self.is_valid_duration(exercise_type, duration):
                vars['counter'] += 1
            vars['in_position'] = False

    def is_valid_duration(self, exercise_type, duration):
        ranges = {
            'squat': (0.5, 2),
            'pushup': (0.5, 2),
            'side_plank': (1, 10),
            'bird_dog': (0.5, 5)
        }
        min_duration, max_duration = ranges.get(exercise_type, (0, float('inf')))
        return min_duration < duration < max_duration

    def draw_landmarks(self, landmarks, frame):
        for landmark in landmarks:
            cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    def display_exercise_info(self, frame, exercise_type):
        vars = self.exercise_vars[exercise_type]
        calories_burned = vars['met_value'] * vars['weight_kg'] * (vars['counter'] / 3600)
        cv2.putText(frame, f"{exercise_type.capitalize()}s: {vars['counter']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Calories: {calories_burned:.2f} kcal", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def start_squat_detection(self):
        self.start_exercise_detection('squat')

    def start_pushup_detection(self):
        self.start_exercise_detection('pushup')

    def start_side_plank_detection(self):
        self.start_exercise_detection('side_plank')

    def start_bird_dog_detection_thread(self):
        detection_thread = Thread(target=self.start_exercise_detection, args=('bird_dog',))
        detection_thread.start()

if _name_ == "_main_":
    root = tk.Tk()
    app = ExerciseDetectionApp(root)
    root.mainloop()
