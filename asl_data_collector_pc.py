"""
ASL Fingerspelling Letter Recognition System - ENHANCED DATA COLLECTOR
Collects static letters only (A-I, K-Y) - 24 letters
J and Z use rule-based motion detection (no training data needed)
"""

from typing import Optional, Dict, Tuple
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
from datetime import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class ASLDataCollector:
    """
    Enhanced ASL data collector with improved logic and user experience
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
        
        # Letters (24 static letters only)
        self.all_letters = list("ABCDEFGHIKLMNOPQRSTUVWXY")
        
        self.current_letter_idx = 0
        self.samples_per_letter = 100
        self.current_samples = 0
        self.letter_completed = False
        
        # Collection mode and state
        self.collection_mode = "all"  # "all" or "single"
        self.collecting = False
        self.countdown = 0
        
        # Progress tracking
        self.progress_file = self.data_dir / "collection_progress.json"
        self.letter_counts = self.load_progress()
        
        # Session statistics
        self.session_start = datetime.now()
        self.session_samples = 0

    def load_progress(self) -> Dict[str, int]:
        """Load existing progress from CSV dataset"""
        dataset_file = self.data_dir / "asl_static_dataset.csv"
        progress = {letter: 0 for letter in self.all_letters}
        
        if dataset_file.exists():
            try:
                df = pd.read_csv(dataset_file)
                if 'label' in df.columns:
                    counts = df['label'].value_counts().to_dict()
                    for letter, count in counts.items():
                        if letter in progress:
                            progress[letter] = count
                    print(f"✓ Loaded progress from existing dataset: {sum(progress.values())} total samples")
            except Exception as e:
                print(f"⚠ Could not load progress: {e}")
        
        return progress

    def save_progress(self):
        """Save progress metadata (optional, for backup)"""
        try:
            progress_data = {
                "letter_counts": self.letter_counts,
                "last_updated": datetime.now().isoformat(),
                "total_samples": sum(self.letter_counts.values())
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠ Could not save progress metadata: {e}")

    def get_current_letter(self) -> Optional[str]:
        """Return the current letter or None if index is out of range"""
        if 0 <= self.current_letter_idx < len(self.all_letters):
            return self.all_letters[self.current_letter_idx]
        return None

    def advance_to_next_letter(self):
        """Advance to next letter in ALL mode"""
        self.current_letter_idx += 1
        if self.current_letter_idx >= len(self.all_letters):
            self.current_letter_idx = 0
        self.current_samples = 0
        self.letter_completed = False
        self.collecting = False
        self.countdown = 0
        
    def jump_to_letter(self, letter: str) -> bool:
        """Jump to a specific letter"""
        if letter not in self.all_letters:
            return False
        
        if self.collecting:
            current = self.get_current_letter()
            print(f"\n⏸  Stopped collecting {current} at {self.current_samples} samples")
            self.collecting = False
        
        self.current_letter_idx = self.all_letters.index(letter)
        self.current_samples = 0
        self.letter_completed = False
        self.countdown = 0
        print(f"\n➡  Switched to letter {letter}")
        return True

    def extract_static_features(self, hand_landmarks) -> Optional[np.ndarray]:
        """Extract 42 features from hand landmarks with improved normalization"""
        if hand_landmarks is None:
            return None
        
        try:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Normalize relative to wrist
            wrist_x, wrist_y = landmarks[0], landmarks[1]
            landmarks[::2] -= wrist_x
            landmarks[1::2] -= wrist_y
            
            # Calculate hand size using multiple reference points for robustness
            # Distance from wrist to middle finger tip
            hand_size = np.sqrt(
                (landmarks[24] - landmarks[0])**2 + 
                (landmarks[25] - landmarks[1])**2
            )
            
            # Add small epsilon to prevent division by zero
            hand_size = max(hand_size, 1e-6)
            
            # Scale by hand size
            landmarks /= hand_size
            
            return landmarks
            
        except Exception as e:
            print(f"⚠ Feature extraction error: {e}")
            return None
    
    def save_datasets(self):
        """Save static dataset as CSV file with proper format (label at end)"""
        if len(self.static_samples) == 0:
            print("\n⚠ No new data to save")
            return

        filepath = self.data_dir / "asl_static_dataset.csv"

        try:
            # Build DataFrame with 42 feature columns + label column at the END
            # This matches the original format where label comes last
            feature_names = [str(i) for i in range(42)]  # Use simple numbers like original
            df_new = pd.DataFrame(self.static_samples, columns=feature_names)
            df_new["label"] = self.static_labels
            
            # Validate data
            if df_new.isnull().any().any():
                print("\n⚠ Warning: Dataset contains null values")
                # Drop rows with nulls
                before = len(df_new)
                df_new = df_new.dropna()
                after = len(df_new)
                if before > after:
                    print(f"   Removed {before - after} rows with null values")
            
            # Merge with existing data
            if filepath.exists():
                try:
                    df_old = pd.read_csv(filepath)
                    
                    # Ensure old data has same column structure
                    if list(df_old.columns) != list(df_new.columns):
                        print(f"\n⚠ Warning: Column mismatch detected")
                        print(f"   Old columns: {len(df_old.columns)}, New columns: {len(df_new.columns)}")
                        
                        # If old data has label at end, it should match
                        if 'label' in df_old.columns:
                            # Reorder old columns to match new format (0-41, label)
                            expected_cols = feature_names + ['label']
                            
                            # Check if old data needs column fixing
                            if set(df_old.columns) != set(expected_cols):
                                print("   Attempting to fix old data format...")
                                # Try to extract just the numeric columns and label
                                label_col = df_old['label']
                                numeric_cols = df_old.select_dtypes(include=[np.number])
                                
                                if len(numeric_cols.columns) == 42:
                                    df_old = numeric_cols.copy()
                                    df_old.columns = feature_names
                                    df_old['label'] = label_col
                                    print("   ✓ Fixed old data format")
                    
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                except Exception as e:
                    print(f"\n⚠ Could not read/merge existing dataset: {e}")
                    print("   Creating new file instead")
                    df_all = df_new
            else:
                df_all = df_new

            # Save to CSV with explicit index=False to avoid index column
            df_all.to_csv(filepath, index=False)
            
            # Update progress tracking
            for letter in df_new['label']:
                self.letter_counts[letter] = self.letter_counts.get(letter, 0) + 1
            self.save_progress()

            print(f"\n✓ Dataset saved successfully: {filepath}")
            print(f"  New samples this session: {len(df_new)}")
            print(f"  Total samples in file: {len(df_all)}")
            print(f"  Features per sample: {len(df_all.columns) - 1}")
            
            # Show per-letter breakdown
            label_counts = df_all['label'].value_counts().sort_index()
            print(f"\n  Samples per letter:")
            for letter in self.all_letters:
                count = label_counts.get(letter, 0)
                status = "✓" if count >= self.samples_per_letter else " "
                print(f"    {status} {letter}: {count}")

            # Clear session buffers
            self.static_samples.clear()
            self.static_labels.clear()
            self.session_samples = 0
            
        except Exception as e:
            print(f"\n❌ Failed to save dataset: {e}")

    def get_next_incomplete_letter(self) -> Optional[str]:
        """Find the next letter that needs more samples"""
        for letter in self.all_letters:
            if self.letter_counts.get(letter, 0) < self.samples_per_letter:
                return letter
        return None

    def draw_ui(self, image: np.ndarray) -> np.ndarray:
        """Draw comprehensive UI overlay"""
        h, w = image.shape[:2]
        current_letter = self.get_current_letter() or "DONE"
        
        # Main status box
        box_height = 280
        cv2.rectangle(image, (10, 10), (w-10, box_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (w-10, box_height), (255, 255, 255), 2)
        
        y_pos = 50
        
        # Current letter (large)
        cv2.putText(image, f"Letter: {current_letter}", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        y_pos += 50
        
        # Letter type and mode
        mode_text = "ALL" if self.collection_mode == "all" else "SINGLE"
        cv2.putText(image, f"Mode: {mode_text} | Type: STATIC", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        y_pos += 40
        
        # Current letter progress
        total_for_letter = self.letter_counts.get(current_letter, 0) + self.current_samples
        cv2.putText(image, f"This letter: {total_for_letter}/{self.samples_per_letter}", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 35
        
        # Session progress
        cv2.putText(image, f"This session: {self.current_samples}", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_pos += 35
        
        # Overall progress
        total_collected = sum(self.letter_counts.values()) + len(self.static_samples)
        total_target = len(self.all_letters) * self.samples_per_letter
        cv2.putText(image, f"Total: {total_collected}/{total_target} ({self.current_letter_idx + 1}/{len(self.all_letters)} letters)", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        
        # Status indicator (top right)
        if self.collecting:
            cv2.circle(image, (w-40, 40), 20, (0, 0, 255), -1)
            cv2.putText(image, "RECORDING", (w-170, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.countdown > 0:
            secs = self.countdown // 30 + 1
            cv2.putText(image, f"Starting in {secs}...", (w-250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        else:
            cv2.circle(image, (w-40, 40), 20, (100, 100, 100), -1)
            cv2.putText(image, "Ready", (w-120, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress bar for current letter
        bar_x, bar_y = 30, box_height - 20
        bar_width = w - 60
        bar_height = 15
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        progress_ratio = min(total_for_letter / self.samples_per_letter, 1.0)
        fill_width = int(bar_width * progress_ratio)
        
        bar_color = (0, 255, 0) if progress_ratio >= 1.0 else (0, 165, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
        
        # Instructions (bottom)
        instructions = [
            "SPACE: Start/Stop | 1-9: Jump to letter | N: Next | S: Save | Q: Quit",
            "Hold hand steady in frame when recording"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(image, text, (30, h - 50 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

    def handle_key_input(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit, True to continue"""
        current_letter = self.get_current_letter()
        
        # Quit
        if key == ord('q') or key == ord('Q'):
            return False
        
        # Toggle collection
        elif key == ord(' '):
            if not self.collecting and self.countdown == 0:
                if self.collection_mode == "single" and self.letter_completed:
                    print("\nℹ Letter already complete. Choose another letter (1-9) or save (S)")
                else:
                    self.countdown = 90  # 3 second countdown
                    print(f"\n▶ Starting collection for {current_letter} in 3...")
            elif self.collecting:
                self.collecting = False
                print(f"\n⏸  Paused at {self.current_samples} samples")
        
        # Skip to next letter (ALL mode only)
        elif key == ord('n') or key == ord('N'):
            if self.collection_mode == "all":
                print(f"\n⏭  Skipped {current_letter}")
                self.advance_to_next_letter()
            else:
                print("\nℹ Next letter (N) only works in ALL mode")
        
        # Save datasets
        elif key == ord('s') or key == ord('S'):
            if len(self.static_samples) > 0:
                self.save_datasets()
            else:
                print("\n⚠ No new data to save")
        
        # Number keys 1-9 for quick letter selection
        elif ord('1') <= key <= ord('9'):
            letter_idx = key - ord('1')
            if letter_idx < len(self.all_letters):
                letter = self.all_letters[letter_idx]
                self.jump_to_letter(letter)
            else:
                print(f"\nℹ Number {letter_idx + 1} doesn't map to a letter")
        
        # Letter keys (0 for 10th letter, etc.)
        elif key == ord('0'):
            if len(self.all_letters) > 9:
                self.jump_to_letter(self.all_letters[9])
        
        return True
    
    def run_collection(self, camera_id: int = 0):
        """Run interactive data collection with improved flow"""
        # Verify camera before starting
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"\n❌ Cannot open camera {camera_id}")
            print("Tips:")
            print("  • Check camera is connected")
            print("  • Try a different camera ID (--camera 1)")
            print("  • Close other apps using the camera")
            return
        
        # Test camera
        ret, test_frame = cap.read()
        if not ret:
            print(f"\n❌ Camera opened but cannot read frames")
            cap.release()
            return
        
        print(f"\n✓ Camera {camera_id} initialized successfully")
        
        # Show collection summary
        print("\n" + "="*70)
        print("ASL DATA COLLECTION - ENHANCED VERSION")
        print("="*70)
        print(f"\nMode: {'ALL letters sequential' if self.collection_mode == 'all' else 'SINGLE letter'}")
        print(f"Starting letter: {self.get_current_letter()}")
        print(f"Samples per letter: {self.samples_per_letter}")
        print(f"\nProgress:")
        
        completed = sum(1 for c in self.letter_counts.values() if c >= self.samples_per_letter)
        print(f"  Completed letters: {completed}/{len(self.all_letters)}")
        print(f"  Total samples: {sum(self.letter_counts.values())}")
        
        if self.collection_mode == "all":
            next_incomplete = self.get_next_incomplete_letter()
            if next_incomplete:
                print(f"\n  Tip: Letter '{next_incomplete}' needs samples")
        
        print("\n" + "="*70)
        
        try:
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("\n⚠ Failed to read frame")
                        break

                    frame = cv2.flip(frame, 1)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw hand landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style())

                            current_letter = self.get_current_letter()
                            if current_letter is None:
                                continue

                            # Collect data
                            if self.collecting and self.countdown == 0:
                                features = self.extract_static_features(hand_landmarks)
                                if features is not None:
                                    self.static_samples.append(features.tolist())
                                    self.static_labels.append(current_letter)
                                    self.current_samples += 1
                                    self.session_samples += 1

                                    # Check completion
                                    if (self.current_samples >= self.samples_per_letter 
                                        and not self.letter_completed):
                                        
                                        print(f"\n✓ Completed {current_letter}: {self.current_samples} samples")
                                        self.letter_completed = True
                                        
                                        if self.collection_mode == "all":
                                            if self.current_letter_idx == len(self.all_letters) - 1:
                                                print("\n🎉 ALL LETTERS COLLECTED!")
                                                self.collecting = False
                                            else:
                                                self.advance_to_next_letter()
                                                self.countdown = 90
                                        else:
                                            self.collecting = False

                    # Handle countdown
                    if self.countdown > 0:
                        self.countdown -= 1
                        if self.countdown == 0:
                            self.collecting = True
                            print(f"▶ Recording {self.get_current_letter()}...")

                    # Draw UI
                    image = self.draw_ui(image)
                    
                    cv2.imshow('ASL Data Collection', image)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # Key was pressed
                        if not self.handle_key_input(key):
                            break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final summary
            duration = (datetime.now() - self.session_start).total_seconds()
            print("\n" + "="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Samples collected: {self.session_samples}")
            
            if self.session_samples > 0:
                print(f"\n⚠ You have {len(self.static_samples)} unsaved samples!")
                print("Press 'S' to save before quitting, or they will be lost.")
                response = input("\nSave now? (y/n): ").strip().lower()
                if response == 'y':
                    self.save_datasets()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced ASL Data Collector - Static Letters (24 letters)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collector.py                    # Default: 100 samples per letter
  python collector.py --samples 200      # Collect 200 samples per letter
  python collector.py --camera 1         # Use camera ID 1
        """
    )
    parser.add_argument('--data-dir', type=str, default='asl_data',
                       help='Directory for saving datasets (default: asl_data)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Target samples per letter (default: 100)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.samples < 10:
        print("⚠ Warning: Very few samples may result in poor model performance")
    elif args.samples > 1000:
        print("⚠ Warning: Many samples will take a long time to collect")
    
    collector = ASLDataCollector(args.data_dir)
    collector.samples_per_letter = args.samples

    # Mode selection
    print("\n" + "="*70)
    print("ASL DATA COLLECTION SETUP")
    print("="*70)
    print("\nCollection Modes:")
    print("  1) ALL - Collect all 24 letters sequentially (A-I, K-Y)")
    print("  2) SINGLE - Focus on one specific letter")
    print("\nRecommendation:")
    print("  • Use ALL for initial dataset creation")
    print("  • Use SINGLE to add more samples for specific letters")
    
    while True:
        mode_choice = input("\nEnter mode (1 or 2): ").strip()
        if mode_choice in ("1", "2"):
            break
        print("Invalid choice. Please enter 1 or 2.")

    if mode_choice == "1":
        collector.collection_mode = "all"
        
        # Suggest starting from incomplete letter
        next_incomplete = collector.get_next_incomplete_letter()
        if next_incomplete:
            idx = collector.all_letters.index(next_incomplete)
            print(f"\nℹ Letter '{next_incomplete}' needs samples. Starting there.")
            collector.current_letter_idx = idx
        
    else:
        collector.collection_mode = "single"
        
        # Show which letters need samples
        incomplete = [l for l in collector.all_letters 
                     if collector.letter_counts.get(l, 0) < collector.samples_per_letter]
        
        if incomplete:
            print(f"\nLetters needing samples: {', '.join(incomplete)}")
        
        while True:
            letter = input(f"\nEnter letter to collect ({', '.join(collector.all_letters)}): ").strip().upper()
            if letter in collector.all_letters:
                collector.current_letter_idx = collector.all_letters.index(letter)
                current_count = collector.letter_counts.get(letter, 0)
                print(f"\nℹ Letter '{letter}' currently has {current_count} samples")
                break
            print(f"Invalid letter. Choose from: {', '.join(collector.all_letters)}")

    # Start collection
    collector.run_collection(args.camera)


if __name__ == '__main__':
    main()