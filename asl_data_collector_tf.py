"""
ASL Fingerspelling Letter Recognition System - DATA COLLECTOR
Collects static letters only (A-I, K-Y) - 24 letters
J and Z use rule-based motion detection (no training data needed)
"""

from typing import Optional
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path


class ASLDataCollector:
    """
    Collect training data for static ASL letters
    Static: A-I, K-Y (24 letters)
    J and Z use rule-based detection (no training needed)
    """
    
    def __init__(self, data_dir: str = "asl_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Data storage
        self.static_samples = []
        self.static_labels = []
        
        # Letters (24 static letters only - J and Z use rule-based detection)
        self.all_letters = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # A-I, K-Y
        
        self.current_letter_idx = 0
        self.samples_per_letter = 100
        self.current_samples = 0
        
    def extract_static_features(self, hand_landmarks) -> Optional[np.ndarray]:
        """Extract 42 features from hand landmarks (21 landmarks × 2 coords)"""
        if hand_landmarks is None:
            return None
        
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Normalize relative to wrist
        wrist_x, wrist_y = landmarks[0], landmarks[1]
        landmarks[::2] -= wrist_x
        landmarks[1::2] -= wrist_y
        
        # Scale by hand size
        hand_size = np.sqrt(
            (landmarks[24] - landmarks[0])**2 + 
            (landmarks[25] - landmarks[1])**2
        ) + 1e-6
        landmarks /= hand_size
        
        return landmarks
    
    def save_datasets(self):
        """Save static dataset as CSV file"""
        if len(self.static_samples) > 0:
            df = pd.DataFrame(self.static_samples)
            df['label'] = self.static_labels
            filepath = self.data_dir / "asl_static_dataset.csv"
            df.to_csv(filepath, index=False)
            print(f"\n✓ Static dataset saved: {filepath}")
            print(f"  Samples: {len(self.static_samples)}")
            print(f"  Features: {len(self.static_samples[0])}")
            print(f"  Letters: {sorted(set(self.static_labels))}")
        else:
            print("\n⚠ No data to save")
    
    def run_collection(self, camera_id: int = 0):
        """Run interactive data collection"""
        cap = cv2.VideoCapture(camera_id)
        
        print("\n" + "="*70)
        print("ASL LETTER DATA COLLECTION - STATIC LETTERS ONLY")
        print("="*70)
        print("\nCollects 24 static letters (A-I, K-Y):")
        print("  • Hold pose steady for each letter")
        print("  • J and Z use rule-based detection (no training needed)")
        print("\nFeatures:")
        print("  • Static: 42 features (21 landmarks × 2 coords)")
        print("\nControls:")
        print("  SPACE: Start/Stop collecting current letter")
        print("  N: Skip to next letter")
        print("  S: Save datasets")
        print("  Q: Quit")
        print("\n" + "="*70 + "\n")
        
        collecting = False
        countdown = 0
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                h, w = image.shape[:2]
                
                # Process hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        
                        current_letter = self.all_letters[self.current_letter_idx] if self.current_letter_idx < len(self.all_letters) else "DONE"
                        
                        # Collect data (static letters only)
                        if collecting and countdown == 0:
                            features = self.extract_static_features(hand_landmarks)
                            if features is not None:
                                self.static_samples.append(features.tolist())
                                self.static_labels.append(current_letter)
                                self.current_samples += 1
                            
                            # Check if done with current letter
                            if self.current_samples >= self.samples_per_letter:
                                print(f"\n✓ Completed {current_letter}: {self.current_samples} samples")
                                self.current_samples = 0
                                self.current_letter_idx += 1
                                
                                if self.current_letter_idx >= len(self.all_letters):
                                    print("\n🎉 ALL LETTERS COLLECTED!")
                                    collecting = False
                                    self.current_letter_idx = 0
                                else:
                                    collecting = False
                                    countdown = 90
                
                # Handle countdown
                if countdown > 0:
                    countdown -= 1
                    if countdown == 0 and self.current_letter_idx < len(self.all_letters):
                        collecting = True
                
                # Draw UI
                current_letter = self.all_letters[self.current_letter_idx] if self.current_letter_idx < len(self.all_letters) else "DONE"
                
                # Status box
                cv2.rectangle(image, (10, 10), (w-10, 220), (0, 0, 0), -1)
                cv2.rectangle(image, (10, 10), (w-10, 220), (255, 255, 255), 2)
                
                # Letter info
                cv2.putText(image, f"Letter: {current_letter}", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                cv2.putText(image, "Type: STATIC", (30, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
                
                cv2.putText(image, "Hold pose steady", (30, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # Progress
                cv2.putText(image, f"Samples: {self.current_samples}/{self.samples_per_letter}", (30, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.putText(image, f"Letter: {self.current_letter_idx + 1}/{len(self.all_letters)}", (30, 195),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Status indicator
                if collecting:
                    cv2.circle(image, (w-40, 40), 20, (0, 0, 255), -1)
                    cv2.putText(image, "HOLD!", (w-120, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif countdown > 0:
                    secs = countdown // 30 + 1
                    cv2.putText(image, f"Starting in {secs}...", (w-250, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                else:
                    cv2.putText(image, "Ready", (w-120, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Instructions
                cv2.putText(image, "SPACE:Start | N:Next | S:Save | Q:Quit", (30, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('ASL Data Collection', image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if not collecting and countdown == 0:
                        countdown = 90  # 3 second countdown
                    elif collecting:
                        collecting = False
                        print(f"\n⏸  Paused at {self.current_samples} samples")
                elif key == ord('n'):
                    if not collecting:
                        print(f"\n⏭  Skipped {current_letter}")
                        self.current_samples = 0
                        self.current_letter_idx += 1
                        if self.current_letter_idx >= len(self.all_letters):
                            self.current_letter_idx = 0
                elif key == ord('s'):
                    if len(self.static_samples) > 0:
                        self.save_datasets()
                    else:
                        print("\n⚠ No data collected yet")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ASL Letter Data Collection - Static Letters Only (24 letters)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='asl_data',
        help='Directory for saving datasets'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Samples per letter'
    )
    
    args = parser.parse_args()
    
    collector = ASLDataCollector(args.data_dir)
    collector.samples_per_letter = args.samples
    collector.run_collection(args.camera)


if __name__ == '__main__':
    main()
