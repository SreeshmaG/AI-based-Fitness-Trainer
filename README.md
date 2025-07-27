

# 🏋️‍♂️ PERSONAL AI GYM - Smart Exercise Detection System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python">
  <img src="https://img.shields.io/badge/OpenCV-4.5+-green?logo=opencv">
  <img src="https://img.shields.io/badge/MediaPipe-0.8-red?logo=mediapipe">
  <img src="https://img.shields.io/badge/Tkinter-GUI-orange?logo=tkinter">

</div>

<br>


## 🌟 Overview

PERSONAL AI GYM is an intelligent fitness assistant that uses computer vision to detect and track your exercises in real-time. With a simple webcam, the system counts your repetitions, tracks workout duration, and estimates calories burned for four fundamental exercises.

## ✨ Key Features

- **Real-time exercise detection** using MediaPipe pose estimation
- **Four supported exercises**: Squats, Push-ups, Side Planks, and Bird-Dogs
- **Automatic rep counting** with form validation
- **Calorie estimation** based on your weight and exercise intensity
- **User-friendly GUI** built with Tkinter
- **Visual feedback** with body landmark tracking

## 🏋️‍♂️ Supported Exercises

| Exercise | Detection Method | Benefits |
|----------|------------------|----------|
| Squats | Hip-knee alignment tracking | Leg strength, core stability |
| Push-ups | Shoulder-elbow position analysis | Upper body strength |
| Side Planks | Body alignment detection | Core strength, balance |
| Bird-Dogs | Limb position tracking | Core stability, back health |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Webcam
- Basic Python packages

### Installation
1. Clone this repository
2. Install required packages:
   ```
   pip install opencv-python mediapipe pillow
   ```
3. Run the application:
   ```
   python personal_ai_gym.py
   ```

## 🛠️ How It Works

1. **Pose Detection**: Uses MediaPipe to identify 33 body landmarks
2. **Exercise Logic**: Specific algorithms for each exercise type
3. **Rep Counting**: Validates movement patterns and counts repetitions
4. **Calorie Calculation**: Estimates calories based on MET values and duration
5. **Visual Feedback**: Displays real-time tracking with landmarks

## 📈 Metrics Tracked

- Repetition count for each exercise
- Estimated calories burned
- Exercise duration
- Form validation

## 🎯 Future Enhancements

- [ ] Add more exercise types
- [ ] Implement user profiles
- [ ] Add form correction feedback
- [ ] Mobile app integration
- [ ] Cloud-based progress tracking


<div align="center">
  <h3>Start your smart fitness journey today! 🚀</h3>
</div>
