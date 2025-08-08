#!/usr/bin/env python3
"""
Script to update existing _keypoints.json files with missing transition_type information.

This script reads existing _contact_analysis.json files to recreate transition data
and updates the corresponding _keypoints.json files with the missing transition_type.

Ideally this script will be only used once. Only works for the specific issue we had
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm


def detect_transitions(contact_log):
    """
    Detect transitions in contact states for each frame.
    """
    left_hand_transitions = []
    right_hand_transitions = []
    
    for i in range(len(contact_log)):
        if i == 0:
            left_hand_transitions.append("no_change")
            right_hand_transitions.append("no_change")
        else:
            prev_left = contact_log[i-1]['left_hand_contact']
            curr_left = contact_log[i]['left_hand_contact']
            prev_right = contact_log[i-1]['right_hand_contact']
            curr_right = contact_log[i]['right_hand_contact']
            
            # Left hand
            if prev_left == False and curr_left == True:
                left_hand_transitions.append("0_to_1")
            elif prev_left == True and curr_left == False:
                left_hand_transitions.append("1_to_0")
            else:
                left_hand_transitions.append("no_change")
            
            # Right hand
            if prev_right == False and curr_right == True:
                right_hand_transitions.append("0_to_1")
            elif prev_right == True and curr_right == False:
                right_hand_transitions.append("1_to_0")
            else:
                right_hand_transitions.append("no_change")
    
    return {
        'left_hand_transitions': left_hand_transitions,
        'right_hand_transitions': right_hand_transitions
    }


def find_closest_future_transitions(transitions, current_frame_idx, total_frames):
    """
    Find the closest future transitions for each hand from the current frame.
    """
    left_transitions = transitions['left_hand_transitions']
    right_transitions = transitions['right_hand_transitions']
    
    result = {
        'left_hand': None,
        'right_hand': None
    }
    
    # Find closest future transition for left hand
    for i in range(current_frame_idx + 1, total_frames):
        if i < len(left_transitions) and left_transitions[i] in ["0_to_1", "1_to_0"]:
            result['left_hand'] = {
                'frame_idx': i,
                'transition_type': left_transitions[i],
                'distance': i - current_frame_idx
            }
            break
    
    # Find closest future transition for right hand
    for i in range(current_frame_idx + 1, total_frames):
        if i < len(right_transitions) and right_transitions[i] in ["0_to_1", "1_to_0"]:
            result['right_hand'] = {
                'frame_idx': i,
                'transition_type': right_transitions[i],
                'distance': i - current_frame_idx
            }
            break
    
    return result


def find_keypoint_transitions_pairs(directory: str, recursive: bool = False) -> List[Tuple[str, str]]:
    """
    Find pairs of _keypoints.json and _transitions.json files.
    """
    pairs = []
    
    if recursive:
        pattern = os.path.join(directory, "**", "*_keypoint_projections_keypoints.json")
        keypoint_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, "*_keypoint_projections_keypoints.json")
        keypoint_files = glob.glob(pattern)
    
    for keypoint_file in keypoint_files:
        # Get the base path by removing "_keypoints.json"
        base_path = keypoint_file.replace("_keypoint_projections_keypoints.json", "")
        transitions_file = f"{base_path}_transitions.json"
        
        if os.path.exists(transitions_file):
            pairs.append((keypoint_file, transitions_file))
        else:
            print(f"Warning: No transitions file found for {os.path.basename(keypoint_file)}")
    
    return pairs


def update_keypoints_file(keypoint_file: str, transitions_file: str, backup: bool = True) -> bool:
    """
    Update a single _keypoints.json file with transition type information.
    """
    try:
        # Load transitions data
        with open(transitions_file, 'r') as f:
            transitions = json.load(f)
        
        # Load keypoints file
        with open(keypoint_file, 'r') as f:
            keypoint_data = json.load(f)
        
        if 'frame_keypoints' not in keypoint_data:
            print(f"No frame_keypoints found in {os.path.basename(keypoint_file)}")
            return False
        
        # Check if already updated
        # if keypoint_data['frame_keypoints']:
        #     first_frame = keypoint_data['frame_keypoints'][0]
        #     if (first_frame.get('keypoint_source_transitions') and 
        #         first_frame['keypoint_source_transitions'].get('left_hand') and 
        #         'transition_type' in first_frame['keypoint_source_transitions']['left_hand']):
        #         print(f"Already updated: {os.path.basename(keypoint_file)}")
        #         return True
        
        # Get total frames from transitions data
        total_frames = len(transitions.get('left_hand_transitions', []))
        final_frame_idx = total_frames - 1
        
        # Update each frame with transition information
        updated_frames = 0
        for frame_data in keypoint_data['frame_keypoints']:
            frame_idx = frame_data['frame_idx']
            
            # Find future transitions
            future_transitions = find_closest_future_transitions(transitions, frame_idx, total_frames)
            
            # Update future_transitions if not already present
            if 'future_transitions' not in frame_data:
                frame_data['future_transitions'] = future_transitions
            
            # Create keypoint_source_transitions
            keypoint_source_transitions = {}
            
            # Left hand
            if future_transitions['left_hand']:
                keypoint_source_transitions['left_hand'] = {
                    'source_frame_idx': future_transitions['left_hand']['frame_idx'],
                    'transition_type': future_transitions['left_hand']['transition_type']
                }
            else:
                keypoint_source_transitions['left_hand'] = {
                    'source_frame_idx': final_frame_idx,
                    'transition_type': 'final_frame'
                }
            
            # Right hand
            if future_transitions['right_hand']:
                keypoint_source_transitions['right_hand'] = {
                    'source_frame_idx': future_transitions['right_hand']['frame_idx'],
                    'transition_type': future_transitions['right_hand']['transition_type']
                }
            else:
                keypoint_source_transitions['right_hand'] = {
                    'source_frame_idx': final_frame_idx,
                    'transition_type': 'final_frame'
                }
            
            frame_data['keypoint_source_transitions'] = keypoint_source_transitions
            updated_frames += 1
        
        # Save updated file
        with open(keypoint_file, 'w') as f:
            json.dump(keypoint_data, f, indent=2, default=str)
        
        print(f"✓ Updated {updated_frames} frames in {os.path.basename(keypoint_file)}")
        return True
        
    except Exception as e:
        print(f"✗ Error updating {os.path.basename(keypoint_file)}: {e}")
        return False


def process_directory(directory: str, backup: bool = True, recursive: bool = False) -> Dict[str, int]:
    """
    Process all _keypoints.json files in a directory.
    """
    print(f"Searching for keypoint files in: {directory}")
    if recursive:
        print("  (searching recursively)")
    
    pairs = find_keypoint_transitions_pairs(directory, recursive)
    
    if not pairs:
        print("No keypoint/transitions file pairs found!")
        return {'total': 0, 'updated': 0, 'failed': 0, 'skipped': 0}
    
    print(f"Found {len(pairs)} keypoint/transitions file pairs")
    print()
    
    stats = {'total': len(pairs), 'updated': 0, 'failed': 0, 'skipped': 0}
    
    for keypoint_file, transitions_file in tqdm(pairs, desc="Updating files"):
        success = update_keypoints_file(keypoint_file, transitions_file, backup)
        if success:
            stats['updated'] += 1
        else:
            stats['failed'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Update _keypoints.json files with missing transition_type information"
    )
    parser.add_argument(
        "directory",
        help="Directory containing _keypoints.json and _contact_analysis.json files"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return 1
    
    print("Transition Type Update Script")
    print("="*50)
    print(f"Directory: {args.directory}")
    print(f"Backup: No")
    print(f"Recursive: Yes")
    print()
    
    stats = process_directory(
        args.directory,
        backup=False,
        recursive=True
    )
    
    print("\n" + "="*50)
    print("UPDATE COMPLETE")
    print("="*50)
    print(f"Total files: {stats['total']}")
    print(f"Updated: {stats['updated']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['total'] > 0:
        success_rate = (stats['updated'] / stats['total']) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if stats['failed'] > 0:
        return 1
    
    print("\nAll files updated successfully! ✓")
    return 0


if __name__ == "__main__":
    exit(main())