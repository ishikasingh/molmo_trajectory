import json
import cv2

def filter_contact_frames(detection_result_path, hand_threshold=0.01, object_threshold=0.98):
    """
    Filter detection results to find valid contact frames.
    A valid contact is when:
    - Hand state is "P" (contact)
    - Hand detection score > hand_threshold  
    - Object detection score > object_threshold
    
    Returns:
        List of tuples: (frame_index, hand_type, hand_bbox, object_bbox)
        where hand_type is either 'left_hand' or 'right_hand'
    """
    detection_result = json.load(open(detection_result_path))
    valid_contacts = []
    
    for i, frame in enumerate(detection_result):
        # Check left hand for contact
        if (frame["left_hand"] is not None and 
            frame["left_hand"]["state"] == "P" and 
            frame["left_hand"]["score"] > hand_threshold and
            frame["objects"] is not None and 
            frame["objects"]["score"] > object_threshold):
            
            valid_contacts.append((
                i, 
                'left_hand', 
                frame["left_hand"]["bbox"], 
                frame["objects"]["bbox"]
            ))
        
        # Check right hand for contact  
        if (frame["right_hand"] is not None and 
            frame["right_hand"]["state"] == "P" and 
            frame["right_hand"]["score"] > hand_threshold and
            frame["objects"] is not None and 
            frame["objects"]["score"] > object_threshold):
            
            valid_contacts.append((
                i, 
                'right_hand', 
                frame["right_hand"]["bbox"], 
                frame["objects"]["bbox"]
            ))
    
    return valid_contacts

def get_contact_statistics(detection_result_path, hand_threshold=0.01, object_threshold=0.98):
    """
    Get statistics about contact frames.
    
    Returns:
        Dictionary with contact statistics
    """
    contacts = filter_contact_frames(detection_result_path, hand_threshold, object_threshold)
    
    left_contacts = [c for c in contacts if c[1] == 'left_hand']
    right_contacts = [c for c in contacts if c[1] == 'right_hand']
    
    return {
        'total_contacts': len(contacts),
        'left_hand_contacts': len(left_contacts),
        'right_hand_contacts': len(right_contacts),
        'contact_frames': [c[0] for c in contacts],  # Frame indices with contacts
        'contact_details': contacts
    }

def get_video_fps(video_path):
    """Get the FPS of the original video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

if __name__ == "__main__":
    detection_result_path = "/home/ANT.AMAZON.COM/fanyangr/code/hand_object_detector/videos/0_video_data.json"
    statistics = get_contact_statistics(detection_result_path)
    print(statistics)
    # left_hand_bbox, right_hand_bbox, object_bbox = extract_bounding_box(detection_result_path)
    # print(left_hand_bbox)
    # print(right_hand_bbox)
    # print(object_bbox)