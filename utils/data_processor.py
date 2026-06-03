"""dataprocessor"""
import threading
import time
import queue
import uuid
import copy
import supervision as sv
import numpy as np
import os, cv2, json
from collections import Counter
from datetime import datetime

from utils.user_preference import SENTENCE_LENGTH_10, SENTENCE_LENGTH_20, SENTENCE_LENGTH_5, AGENT_NAME
from utils.user_preference import UserPreference
from utils.worldscribe_utils import encode_image_to_base64, get_caption_from_yolo_classes_adv, get_caption_from_yolo_classes, base64_to_cv2_image
from utils.firebase.firebase_manager import FirebaseWriteManager, angle_difference
from utils.firebase.firestore_manager import FirestoreManager
from utils.nlp_process import concise_sentence, get_sentence_similarity_spacy
from utils.speech import TTSManager
from utils.classes import COCO_CLASSES
from ultralytics import YOLO
from similarity.VGG16_similarity_Xing import get_frame_similarity

SOUND_CONFIDENCE_THRES = 0.3 # sound confidence
SIMILARITY_THRESHOLD = 0.5 # similarity threshold
TURN_DEGREE_THRESHOLD = 30 # the descriptions stop if the user turns more than 30 degree
MESSY_ELEMENT = [99999999, 99999999, 99999999]

def count_differences(list1, list2):
    set1, set2 = set(list1), set(list2)
    diff_count = len(set1.symmetric_difference(set2))  # Elements not in both lists
    return diff_count

def has_n_differences(list1, list2, n):
    return count_differences(list1, list2) >= n


class DataProcessor:
    def __init__(self, use_mask=False):
        # ignore this, this is used to calculate the average time of YOLO, which was reported in the paper
        # self.yolo_num = 0
        # self.yolo_time = 0

        # logging each user's data everytime the program is run
        self.firestoreManager = FirestoreManager()
        

        # tracking live tts threads and stop them if needed
        self.live_tts_threads = []
        self.stop_threads_event = threading.Event()

        # This is YOLO model and annotators for visualization (e.g., bounding boxes), which will be determined later in process_data_by_yolo
        self.model = None
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.frame_id = 0
        self.frame_history = {}
        self._caption_queue = "start"
        self._lock = threading.Lock()

        # track which sounds are happening
        self.sound_event_track = {}
        # track which sound manipulations are being applied
        self.ongoing_sound_manipulation = {}
        # This stores the user's preference on their smartphone
        self.user_preference = UserPreference(self)

        # store the current frame_info
        self.frame_info = None

        # record time when the system starts
        self.start_time = 0
        # The current spoken descriptions
        self.current_caption = ""
        # This manager writes and retreives data from the firebase
        self.firebaseWriteManager = FirebaseWriteManager()
        self.timestamp = self.firebaseWriteManager.ref_user_timestamp.get()
        # This manager supports text to speech and deal with the interruption
        self.ttsManager = TTSManager(self.firebaseWriteManager)

        self.last_time_speak = 0
        self.user_current_clock = 0
        self.user_current_degree = None
        self.user_prev_degree = None
        self.is_talking_less = False
        self.num_frame_to_check = 6

        self.prev_read_yolo_caption = None


        # The number of frames that does not have consistent object composition in the scene
        # self.idle_frame_num = 0
        self.prev_scene_state = {
            "state": None,
            "sources_used": [],
            "descriptions": [],
            "frame_id": 0,
            "uuid": None
        }
        self.curr_scene_state = self.prev_scene_state

        # Simulate short-term memory that do not overlap the previous descriptions
        self.caption_history = []

        # description can be stopped due to either the user's manipulation or they turn around over 30 degree
        self.stop_streaming_reason = ""

        self.ids_stable_state = list(range(10000, 10100))
        self.prev_state_details = {
                "thread": None,
                "state" : None,
                "ids"   : self.ids_stable_state,
                "frame" : None,
                "frame_id" : self.frame_id, 
                "state_changed_time" : time.time()
            }
        self.timer = None

    # The generated descriptions will be put into the caption queue and retrieved later
    @property
    def caption_queue(self) -> str:
        with self._lock:
            return self._caption_queue

    @caption_queue.setter
    def caption_queue(self, value: str):
        with self._lock:
            self._caption_queue = value
    
    def process_audio_data(self, server_data_dict):
        # this function is called in firebase_manager.py
        # it is only called if there are sounds detected by the smartphone

        # we get the data, such as label, confidence and timestamp from the smartphone
        label        = server_data_dict['label'] 
        confidence   = server_data_dict['confidence'] 
        timestamp    = int(server_data_dict['timestamp'])

        
        print(f"volume: {self.ttsManager.volume_factor} --- Speed: {self.ttsManager.speech_speed} --- {label} --- {confidence} --- {self.ongoing_sound_manipulation}")
        # tracking the sound names
        if label not in self.sound_event_track: 
            self.sound_event_track[label] = [] # create a new key using the name
        # insert the instance of confidence and timestamp of the data
        self.sound_event_track[label].insert(0, (confidence,timestamp))
        # maintain the first 30 data points in the dict        
        self.sound_event_track[label] = self.sound_event_track[label][:30]

        # make sure the sounds are happening instead of false positives by checking the first five instances of the sound label
        is_happening = self.check_sound_happening(label)
        # make sure three things: 
        # 1. the sound is happening. 
        # 2. the user has specified the manipulation. 
        # 3. the sound manipulation for the sound is not being applied
        if is_happening and \
            label in self.user_preference.manipulation_table and \
            label not in self.ongoing_sound_manipulation:
            print(f'{label} is happening...')
            
            # extract the manipulation mapping (customized by users on the mobile interface)
            manipulation = self.user_preference.manipulation_table[label]
            
            # increase the volume 
            if manipulation == 'volume_high' and not self.ttsManager.high_volume:
                threading.Thread(target=self.ttsManager.increase_volume_and_back, args=()).start()
            
            # pause the descriptions
            elif manipulation == 'pause':
                self.provide_reason_to_stop("manipulation")
                # self.ttsManager.stop_streaming()
                # self.stop_streaming_reason = "manipulation"

            # add the sound to the list of ongoing manipulated sounds
            self.ongoing_sound_manipulation[label] = manipulation
        
        # remove those sounds that do not happen anymore
        self.check_ongoing_sound_manipulation()

        return 
    

    def check_sound_happening(self, label: str) -> bool:
        # We don't consider the number of instances of a sound label less than 5
        if len(self.sound_event_track[label]) < 5:
            return False
        # if all the first five confidences of a sound label are more than the threshold, then return true
        return all(conf >= SOUND_CONFIDENCE_THRES for conf, _ in self.sound_event_track[label][:5])

    def check_sound_ending(self, label: str) -> bool:
        # check if the sound is ending by the first five instances of the sound label
        return all(conf < SOUND_CONFIDENCE_THRES for conf, _ in self.sound_event_track[label][:5])
    

    def check_ongoing_sound_manipulation(self) -> None:
        """
        Remove the manipulations by checking if each sound is still happening in current give instances
        """
        labels_to_delete = []
        instances_to_check = 5
        for label, manipulation in self.ongoing_sound_manipulation.items():
            if len(self.sound_event_track[label]) < instances_to_check or not self.check_sound_ending(label):
                continue

            print(f"{label} is ending...")
            self.reset_sound_manipulation(manipulation)
            labels_to_delete.append(label)

        for label in labels_to_delete:
            del self.ongoing_sound_manipulation[label]
    
    def reset_sound_manipulation(self, manipulation: str) -> None:
        """
        Return to the original sound settings
        """
        if manipulation == "volume_high" and self.ttsManager.high_volume:
            threading.Thread(target=self.ttsManager.increase_volume_and_back).start()
        elif manipulation == "pause":
            # self.ttsManager.start_streaming()
            self.stop_streaming_reason = ""
            


    def initialize_frame_info(self, frame, server_data_dict):
        """
        Initializes frame information dictionary with metadata and placeholders.
        """
        return {
            "frame_id": self.frame_id,
            "frame": frame,
            "frame_shape": frame.shape,
            "boxes": [],
            "ids": [],
            "object_classes": [],
            "images": [],
            "masks": [],
            "confs": [],
            "yolo_caption": None,
            "text": None,
            "uuid": str(uuid.uuid4().hex),
            "frame_base64": server_data_dict.get('frame_base64', ''),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_s": time.time()
        }


    def process_data_by_yolo(self, frame, server_data_dict):
        """
        Processes a single frame using YOLO for object detection, tracks objects, 
        and handles annotations and captions.
        """
        # Validate user preference for custom classes
        # self.user_preference.custom_classes = COCO_CLASSES
        if not self.user_preference.custom_classes:
            return None

        # Initialize YOLO model if not already initialized
        if self.model is None:
            # self.model = YOLO('./../models/yolov8s-worldv2.pt') # TODO: change if there is any problems
            # self.model.set_classes(self.user_preference.custom_classes)
            self.model = YOLO('models/yolo11n.pt') 
            # self.model.set_classes(COCO_CLASSES)

        # Set up frame processing information
        if self.start_time == 0:
            self.start_time = time.time()
        self.frame_id += 1
        frame_info = self.initialize_frame_info(frame, server_data_dict)

        # Run YOLO model inference
        # start_time = time.time()
        results = self.model.track(frame, persist=True, verbose=False, conf=0.4, device="cpu")
        # self.yolo_time += time.time() - start_time
        # self.yolo_num += 1
        # print(f"YOLO avg time: {self.yolo_time / self.yolo_num}")
        

        # Update frame information with YOLO results
        self.update_frame_info_with_results(frame_info, results)

        # Generate and store caption
        self.generate_caption_for_frame(frame_info, server_data_dict)

        # update the goal frame_info
        self.frame_info = frame_info

        # Start background processing for new captions
        if self.frame_id%8 ==0:
            state, sentence_requirement = self.update_scene_state(self.frame_id, frame_info['uuid'])
            # print(f"{'*'*40} 123 {state} {'*'*40}")
            if state:
                print(f"{'*'*40} {state} {'*'*40}")
                self.prev_scene_state = self.curr_scene_state
                self.curr_scene_state["state"] = state
                self.curr_scene_state["frame_id"] = self.frame_id
                self.curr_scene_state["uuid"] = frame_info["uuid"]
                self.curr_scene_state["sources_used"] = []
                self.curr_scene_state["descriptions"] = []

                if self.curr_scene_state["state"]:
                    print(f"{'*'*40} {state} {'*'*40}")
                    print(f"{'*'*40} {state} {'*'*40}")
                    print(f"{'*'*40} {state} {'*'*40}")
                    print(f"{'*'*40} {state} {'*'*40}")
                    self.provide_reason_to_stop('turn')
                    self.describe_for_new_scene(self.frame_id, frame_info['uuid'], state, sentence_requirement)
        # Handle TTS streaming
        self.tts_streaming()

        # Annotate frame for visualization/debugging
        detections = sv.Detections.from_ultralytics(results[0])
        annotated_frame = self.annotate_frame(frame, detections)
        # print(f"------------------ FPS: {self.frame_id / (time.time() - self.start_time)} ------------------")
        return annotated_frame

    def update_frame_info_with_results(self, frame_info, results):
        """
        Updates the frame information dictionary with YOLO model results.
        """

        # print("self.user_preference.custom_classes: ", len(self.user_preference.custom_classes), self.user_preference.custom_classes)
        if results[0].boxes.id is not None:
            frame_info.update({
                "boxes": np.array(results[0].boxes.xyxy.cpu()),
                "ids": results[0].boxes.id.int().cpu().tolist(),
                "confs": results[0].boxes.conf.cpu().tolist(),
                "object_classes": [
                    self.user_preference.custom_classes[cls_num]
                    for cls_num in np.array(results[0].boxes.cls).astype(int)
                ]
            })

    def generate_caption_for_frame(self, frame_info, server_data_dict):
        """
        Generates a caption for the current frame and stores it in the frame history.
        """
        # yolo_caption = get_caption_from_yolo_classes(frame_info, self.user_preference.object_preference)
        yolo_caption = get_caption_from_yolo_classes(frame_info["object_classes"])
        frame_info['yolo_caption'] = [{
            "caption": yolo_caption,
            "frame_id": self.frame_id,
            "event": "default_caption",
            "state_changed_time": time.time(),
            "completion_time": time.time(),
            "uuid": frame_info["uuid"],
            "user_degree": self.user_current_degree,
            "payload": None,
            "frame_base64": server_data_dict.get('frame_base64', ''),
            "ids": frame_info["ids"],
            "system_role": None,
            "user_msg": None,
            "user_goal": self.user_preference.user_goal,
            "sentence_requirement": None,
            "clss": frame_info["object_classes"],
            "similarity_score": None,
            "depth_score": None,
            "adj_preference": self.user_preference.adj_preference,
            "source": "yolo"
        }]
        self.frame_history[self.frame_id] = frame_info

    def annotate_frame(self, frame, detections):
        """
        Annotates a frame with bounding boxes and labels for visualization.
        """
        labels = [
            f"{self.user_preference.custom_classes[class_id]} {confidence:.3f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        return self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    
    def is_user_turn(self):
        
        if self.user_current_degree is None: 
            self.user_current_degree = self.firebaseWriteManager.get_user_degree()
        if self.user_prev_degree is None: 
            self.user_prev_degree = self.firebaseWriteManager.get_user_degree()
        
        diff = angle_difference(self.user_current_degree, self.user_prev_degree)

        if diff >= TURN_DEGREE_THRESHOLD:
            self.user_prev_degree = self.user_current_degree
            return True
        else: 
            return False


    def update_scene_state(self, frame_id, frame_uuid):
        if frame_id <= self.num_frame_to_check:
            return None, None
        return self.determine_scene_state(frame_id)
        # if state:
        #     self.describe_for_new_scene(frame_id, _uuid, state, sentence_requirement)

    def determine_scene_state(self, frame_id):
        """
        Determines the current scene state based on frame history and user actions.
        """
        
        recent_ids = self.get_recent_ids(frame_id, self.num_frame_to_check)
        # print(f"frame_id: {str(frame_id)}, recent_ids: {str(recent_ids)}")
        frame = self.frame_history[frame_id]['frame']

        # Initialize defaults.
        state = ""
        sentence_requirement = SENTENCE_LENGTH_10

        # state can be user_turn, new_scene, messy_scene, empty_scene
        if self.is_user_turn():
            state = "new_scene"

        elif self.is_new_scene(recent_ids, frame_id):
            # print("testing new scene....")
            state = self.handle_long_scene(recent_ids, frame_id, frame, "new_scene", "long_new_scene")

        elif self.is_empty_scene(recent_ids, frame_id):
            # print("testing empty scene....")
            state = self.handle_long_scene(recent_ids, frame_id, frame, "empty_scene", "long_empty_scene")

        # elif self.is_messy_scene(recent_ids, frame_id):
        #     print("testing messy scene....")
        #     state = self.handle_long_scene(recent_ids, frame_id, frame, "messy_scene", "long_messy_scene")

        if state and self.user_preference.granularity_preference['adaptive'] == "true":
            sentence_requirement = self.adapt_sentence_length(state, frame_id)
        elif state:
            sentence_requirement = self.get_sentence_length_preference()

        return state, sentence_requirement

    def describe_for_new_scene(self, frame_id, _uuid, state, sentence_requirement):
        """
        Handles transitions to new caption states, including updates and server interactions.
        """
        print(f"$$$$$$$$$$$$$sending description request for frame {frame_id} and state {state}........")
        system_role = self.user_preference.create_systemrole(sentence_requirement)
        user_msg = self.user_preference.create_user_requirement(sentence_requirement)

        state_changed_time = time.time()
        # self.firebaseWriteManager.play_audio(state) 

        self.prev_state_details = {
            "event": state,
            "ids": self.ids_stable_state,
            "clss": self.frame_history[frame_id]['object_classes'],
            "frame": self.frame_history[frame_id]['frame'],
            "frame_id": frame_id,
            "state_changed_time": state_changed_time,
            "uuid": _uuid,
            "system_role": system_role,
            "user_msg": user_msg,
            "user_goal": self.user_preference.user_goal,
            "sentence_requirement": sentence_requirement,
            "user_degree": self.user_current_degree,
            "user_goal_type": self.user_preference.user_goal_type,
        }

        threading.Thread(target=self.generate_descriptions, args=(copy.deepcopy(self.prev_state_details),)).start()

    def generate_descriptions(self, prev_state_details):
        """
        Sends frame information to an external server for caption generation.
        """
        prev_state_details['timestamp'] = time.time()
        prev_state_details['frame'] = encode_image_to_base64(prev_state_details['frame'])
        self.firebaseWriteManager.send_image_to_server(prev_state_details)
        return prev_state_details
    
    
    def has_n_repeated_elements(self, lst, n):
        """
        Check if any sublist in the list appears at least n times.

        Parameters:
            lst (list): The list to check, which can contain sublists.
            n (int): The number of repetitions to check for.

        Returns:
            tuple: (bool, element) where bool indicates if a repeated sublist exists, 
                and element is the repeated sublist (or None if not found).
        """
        # Convert sublists to tuples for hashing
        hashable_list = [tuple(sublist) if isinstance(sublist, list) else sublist for sublist in lst]
        
        # Count occurrences
        from collections import Counter
        element_counts = Counter(hashable_list)

        # Find an element repeated at least n times
        for element, count in element_counts.items():
            if count >= n:
                # Convert back to list if the element was originally a sublist
                element = list(element)
                element = sorted(element)
                return True, element

        return False, None
    
    def has_over_n_distinct_elements(self, input_list, N, m):
        """
        Checks if there are more than N distinct elements in the list, 
        where each distinct element appears more than m times.

        Args:
            input_list (list): A list of elements (elements may be lists).
            N (int): The threshold for the number of distinct elements.
            m (int): The minimum number of occurrences for an element to count as distinct.

        Returns:
            bool: True if there are more than N distinct elements that appear more than m times, False otherwise.
        """
        from collections import Counter

        # Convert all elements to tuples (if they are lists) to make them hashable
        hashable_elements = [tuple(elem) if isinstance(elem, list) else elem for elem in input_list]
        # Count occurrences of each element
        element_counts = Counter(hashable_elements)
        # Filter elements that appear more than m times
        frequent_elements = [elem for elem, count in element_counts.items() if count >= m]

        # Check if the number of distinct frequent elements exceeds N
        return len(frequent_elements) >= N

        
    def is_new_scene(self, recent_ids, frame_id):
        """
        Checks if the current scene is a new scene.
        """
        repeated , element = self.has_n_repeated_elements(recent_ids, len(recent_ids)-1) 
        return (
            repeated and len(element) > 0 and
            frame_id - self.prev_state_details['frame_id'] >= self.num_frame_to_check
        )

    def is_empty_scene(self, recent_ids, frame_id):
        """
        Checks if the current scene is an empty scene.
        """
        repeated , element = self.has_n_repeated_elements(recent_ids, len(recent_ids)-1) 
        return (
            repeated and len(element) == 0 and
            frame_id - self.prev_state_details['frame_id'] >= self.num_frame_to_check
        )


    def is_messy_scene(self, recent_ids, frame_id):
        """
        Checks if the current scene is a messy scene.
        """
        if self.has_over_n_distinct_elements(recent_ids, 2, 2) or self.has_over_n_distinct_elements(recent_ids, 4, 1):
            return True
        return False


    def handle_long_scene(self, recent_ids, frame_id, frame, scene_state, long_scene_state):
        """
        Handles both new and long scene detection logic.
        """
        # long_num_frame_to_check = 24
        recent_repeated , recent_element = self.has_n_repeated_elements(recent_ids, len(recent_ids)-1) 
        # long_ids = self.get_recent_ids(frame_id, long_num_frame_to_check)
        print(f"ids_stable_state: {self.ids_stable_state}, {recent_repeated}, recent_element: {recent_element}")
        
        if scene_state == "empty_scene":
            if self.prev_state_details["frame"] is not None and frame is not None:
                similarity = get_frame_similarity(self.prev_state_details["frame"], frame)
                print(f"{'$'*20} similarity score {similarity[0][0]} {'$'*20}")
                if similarity[0][0] > 0.85: 
                    return None
                else:
                    return scene_state
            
        
        elif scene_state == "new_scene":
            if recent_repeated and recent_element != self.ids_stable_state:
                
                if has_n_differences(recent_element, self.ids_stable_state, 3):
                    # if new_scene_state == "messy_scene": self.ids_stable_state = MESSY_ELEMENT
                    self.ids_stable_state = recent_element
                    return scene_state
                else: 
                    # self.ids_stable_state = recent_element
                    return None
            else:
                return None
        # else:
        #     return None


    def adapt_sentence_length(self, state, frame_id):
        """
        Adapts sentence length based on scene state and user preference.
        """
        if 'long' in state:
            return SENTENCE_LENGTH_20
        elif len(self.frame_history[frame_id]['ids']) <= 5:
            return SENTENCE_LENGTH_10
        return SENTENCE_LENGTH_5

    def get_sentence_length_preference(self):
        """
        Returns the sentence length based on user preferences.
        """
        preference = self.user_preference.granularity_preference
        if preference['verbose'] == 'true':
            return SENTENCE_LENGTH_20
        elif preference['normal'] == 'true':
            return SENTENCE_LENGTH_10
        elif preference['concise'] == 'true':
            return SENTENCE_LENGTH_5
        return SENTENCE_LENGTH_10

    def get_recent_ids(self, frame_id, count):
        """
        Retrieves the recent IDs from frame history.
        """
        return [self.frame_history[i]['ids'] for i in range(frame_id - count, frame_id)] if frame_id - count > 0 else None


    def get_caption_for_tts(self):
        """
        Retrieves and processes the appropriate caption for text-to-speech (TTS) streaming.
        """
        # Retrieve caption queues from different models
        selected_source = None
        selected_caption_info = None

        if self.curr_scene_state["state"] == "user_turn":
            selected_source = "yolo"
            selected_caption_info = self.frame_history[self.frame_id].get("yolo_caption", [{}])[0] 
            # print(f"{'^'*40}user_turn selected_caption " , selected_source)
            if not selected_caption_info['caption']:
                selected_source = "moondream"
                selected_caption_info = self.firebaseWriteManager.get_moondream_caption_buffer()
                # print(f"{'^'*40}user_turn selected_caption " , selected_source)

                # print("user_turn moondream selected_caption_info", selected_caption_info['caption'])

        elif self.curr_scene_state["state"] == "new_scene":
            print(f"{'*'*80}")
            print(f"{'*'*80}")

            #################### comment out below ####################
            # index = self.curr_scene_state['sources_used'].count('gpt')
            # try:
            #     temp = self.firebaseWriteManager.get_caption_buffer()[index]
                
            #     if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
            #         return
            #     print("new_scene*******check gpt", temp['frame_id'], self.prev_state_details['frame_id'])
            #     if temp['frame_id'] >= self.prev_state_details['frame_id']:
            #         self.curr_scene_state['sources_used'].append("gpt")
            #         selected_source = "gpt"
            #         selected_caption_info = temp
            #     # print(f"{'^'*40} {selected_caption_info.keys()}")                    
            # except Exception as e:
            #     print("An error occurred:", e)
            #     selected_source = None
            #     self.curr_scene_state['sources_used'].clear()
            
            # if "gpt" not in self.curr_scene_state['sources_used']:
            #     temp = self.firebaseWriteManager.get_moondream_caption_buffer()[0]
            #     # print("temp:::::::::", temp)
            #     print(f"new_scene*******check moondream1")
            #     if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
            #         return
            #     print(f"new_scene*******check moondream2", temp['frame_id'], self.prev_state_details['frame_id'])
            #     if temp['frame_id'] >= self.prev_state_details['frame_id']:
            #         # print("new_scene*******check moondream", temp['frame_id'], self.prev_state_details['frame_id'])
            #         self.curr_scene_state['sources_used'].append("moondream")
            #         selected_source = "moondream"
            #         selected_caption_info = temp
            #################### comment out above ####################

            #################### and resume below ####################
            if "moondream" not in self.curr_scene_state['sources_used']:
                temp = self.firebaseWriteManager.get_moondream_caption_buffer()[0]
                # print("temp:::::::::", temp)
                print(f"new_scene*******check moondream1")
                if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
                    return
                print(f"new_scene*******check moondream2", temp['frame_id'], self.prev_state_details['frame_id'])
                if temp['frame_id'] >= self.prev_state_details['frame_id']:
                    # print("new_scene*******check moondream", temp['frame_id'], self.prev_state_details['frame_id'])
                    self.curr_scene_state['sources_used'].append("moondream")
                    selected_source = "moondream"
                    selected_caption_info = temp
            

            elif "moondream" in self.curr_scene_state['sources_used']:
                print(f"new_scene*******check gpt1")

                index = self.curr_scene_state['sources_used'].count('gpt')
                try:
                    temp = self.firebaseWriteManager.get_caption_buffer()[index]
                    
                    if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
                        return
                    print("new_scene*******check gpt", temp['frame_id'], self.prev_state_details['frame_id'])
                    if temp['frame_id'] >= self.prev_state_details['frame_id']:
                        self.curr_scene_state['sources_used'].append("gpt")
                        selected_source = "gpt"
                        selected_caption_info = temp
                    # print(f"{'^'*40} {selected_caption_info.keys()}")                    
                except Exception as e:
                    print("An error occurred:", e)
                    selected_source = None
                    self.curr_scene_state['sources_used'].clear()
            #################### and resume above ####################
                    
        elif self.curr_scene_state["state"] == "empty_scene":
            print(f"{'#'*80}")
            print(f"{'#'*80}")
            #################### comment out below ####################
            # index = self.curr_scene_state['sources_used'].count('gpt')
            # # print(f"{'^'*40}new_scene selected_caption " , selected_source, index)
            # try:
            #     temp = self.firebaseWriteManager.get_caption_buffer()[index]
            #     if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
            #         return
            #     print("empty_scene*******check gpt", temp['frame_id'], self.prev_state_details['frame_id'])
            #     if temp['frame_id'] >= self.prev_state_details['frame_id']:
            #         self.curr_scene_state['sources_used'].append("gpt")
            #         selected_source = "gpt"
            #         selected_caption_info = temp
            #     # print(f"{'^'*40} {selected_caption_info.keys()}")                    
            # except Exception as e:
            #     print("An error occurred:", e)
            #     selected_source = None
            #     self.curr_scene_state['sources_used'].clear()
            # if "gpt" not in self.curr_scene_state['sources_used']:
            #     print(f"new_scene*******check moondream")
            #     temp = self.firebaseWriteManager.get_moondream_caption_buffer()[0]
            #     if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
            #         return
            #     print("empty_scene*******check moondream", temp['frame_id'], self.prev_state_details['frame_id'])
            #     if temp['frame_id'] >= self.prev_state_details['frame_id']:
            #         self.curr_scene_state['sources_used'].append("moondream")
            #         selected_source = "moondream"
            #         selected_caption_info = temp
            #################### comment out above ####################

             #################### and resume below ####################
            if "moondream" not in self.curr_scene_state['sources_used']:
                print(f"new_scene*******check moondream")
                temp = self.firebaseWriteManager.get_moondream_caption_buffer()[0]
                if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
                    return
                print("empty_scene*******check moondream", temp['frame_id'], self.prev_state_details['frame_id'])
                if temp['frame_id'] >= self.prev_state_details['frame_id']:
                    self.curr_scene_state['sources_used'].append("moondream")
                    selected_source = "moondream"
                    selected_caption_info = temp
                # print(f"{'^'*40}empty_scene selected_caption " , selected_source)
            if "moondream" in self.curr_scene_state['sources_used']:
                print(f"new_scene*******check gpt")
                index = self.curr_scene_state['sources_used'].count('gpt')
                # print(f"{'^'*40}new_scene selected_caption " , selected_source, index)
                try:
                    temp = self.firebaseWriteManager.get_caption_buffer()[index]
                    if temp in ['s', 't', 'a', 'r', 't'] or temp == "start":
                        return
                    print("empty_scene*******check gpt", temp['frame_id'], self.prev_state_details['frame_id'])
                    if temp['frame_id'] >= self.prev_state_details['frame_id']:
                        self.curr_scene_state['sources_used'].append("gpt")
                        selected_source = "gpt"
                        selected_caption_info = temp
                    # print(f"{'^'*40} {selected_caption_info.keys()}")                    
                except Exception as e:
                    print("An error occurred:", e)
                    selected_source = None
                    self.curr_scene_state['sources_used'].clear()
            #################### and resume above ####################


        # Fallback to YOLO captions if no other caption queue is selected
        if not selected_source:
            # return None
            selected_source = "yolo"
            selected_caption_info = self.frame_history[self.frame_id].get("yolo_caption", [{}])[0]

            if selected_caption_info['caption'] != self.prev_read_yolo_caption:
                self.prev_read_yolo_caption = selected_caption_info['caption']
            else:
                return None
                


        if not selected_caption_info or isinstance(selected_caption_info,list) or isinstance(selected_caption_info,str):
            print("No valid caption found.")
            return None

        caption = selected_caption_info["caption"]
        self.curr_scene_state['descriptions'].append(caption)
        # print(f"{'^'*100}")
        # print(f"{'^'*100}")
        # print(f"{'^'*40}caption:", caption)
        # print(f"{'^'*100}")
        # print(f"{'^'*100}")

        selected_caption_info['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        selected_caption_info['timestamp_s'] = time.time()
        print("selected_source: ", selected_source)
        print("selected_source['caption']: ", selected_caption_info['caption'])

        if selected_source == "yolo":
            self.logging("yolo", selected_caption_info)
        try:
            self.firestoreManager.read_caption_update(selected_caption_info)
        except:
            print("[ERROR] when logging data to firestore")
        return caption

    def logging(self, source, selected_caption_info):
        
        # Create the folder structure
        print(selected_caption_info.keys())
        image = selected_caption_info['frame_base64']
        image = base64_to_cv2_image(image)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        subfolder = source + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join("ASSETS2025", self.timestamp, subfolder)
        os.makedirs(folder, exist_ok=True)  # Ensure folder exists

        # Save the image
        image_address = os.path.join(folder, f"{subfolder}.jpg")
        selected_caption_info['image_address'] = image_address
        selected_caption_info['json_address']  = os.path.join(folder, f"{subfolder}.json")
        selected_caption_info['filename']      = subfolder
        cv2.imwrite(image_address, image)  # Save the image file

        # Save the JSON log
        json_address = os.path.join(folder, f"{subfolder}.json")
        with open(json_address, "w", encoding="utf-8") as json_file:
            json.dump(selected_caption_info, json_file, ensure_ascii=False, indent=4)

        print(f"Saved image: {image_address}")
        print(f"Saved JSON log: {json_address}")
            
        

    def select_best_caption(self, caption_queues):
        """
        Selects the best caption from available caption queues based on similarity and state.
        """
        for queue_name, caption_queue in caption_queues.items():
            if not self.is_valid_caption_queue(caption_queue):
                continue

            caption_info = caption_queue[0]

            # Check UUID and other attributes for selection criteria
            if self.is_matching_uuid(caption_info) or self.is_similar_to_previous_frame(caption_info):
                # print(f"\n\nSelected caption from {queue_name}. \n\n{caption_info}")
                return queue_name, caption_info

        return None, None

    def is_valid_caption_queue(self, caption_queue):
        """
        Validates the caption queue to ensure it contains valid data.
        """
        return caption_queue not in [None, "start", ["s", "t", "a", "r", "t"]]

    def is_matching_uuid(self, caption):
        """
        Checks if the caption UUID matches the previous state UUID.
        """
        return caption.get("uuid") == self.prev_state_details["uuid"]

    def is_similar_to_previous_frame(self, caption):
        """
        Checks if the caption corresponds to a frame similar to the previous state frame.
        """
        similarity = get_frame_similarity(
            self.prev_state_details["frame"],
            self.frame_history[caption["frame_id"]]["frame"]
        )
        return similarity > SIMILARITY_THRESHOLD

    def manipulate_content(self, caption):
        """
        Modifies the caption content based on user preferences and syntax corrections.
        """
        if not caption:
            return None

        if self.is_talking_less:
            caption = concise_sentence(caption)

        return caption.replace(" ,", ",").replace(" .", ".").replace("  ", " ")


    def filter_caption_queue(self, caption_queue):
        """
        Checking the previous sentences and avoiding repeating
        """
        if len(caption_queue) == 0: return None
        if caption_queue[0]['source'] == "yolo": 
            return caption_queue[0]

        caption_history = [ (item['caption'],item['uuid']) for item in self.caption_history]
        
        d = 3 # set the number of previous descriptions to check
        index = len(self.caption_history) if len(self.caption_history) < d else d
        caption_latest = [item['caption'] for item in self.caption_history[-index:]]

        for item in caption_queue:
            if (item['caption'], item['uuid']) not in caption_history:            
                if item['caption'] not in caption_latest:
                    if len(self.caption_history) < d:
                        self.caption_history.append(item)
                        return item
                    else:
                        scores = get_sentence_similarity_spacy([item['caption']], caption_latest)
                        if max(scores) < 0.8:
                            self.caption_history.append(item)
                            return item
        return None
    
    def provide_reason_to_stop(self, reason):
        
        dont_interrupt = [
            "invoke_agent",
            "waiting_for_user_query_and_answer"
        ]
        if self.stop_streaming_reason != reason:
            if reason == "invoke_agent":
                self.stop_streaming_reason = reason
                return "invoke_agent"
            elif reason == "turn" and self.stop_streaming_reason not in dont_interrupt:
                self.stop_streaming_reason = reason
                return "turn"
    
    def start_timer(self, timeout, callback):
        """Starts a new timer and cancels the previous one if it exists."""
        # Cancel the existing timer if it's running
        if self.timer is not None:
            self.timer.cancel()

        # Create a new timer
        self.timer = threading.Timer(timeout, callback)
        self.timer.start()
    
    def on_timeout(self):
        self.ttsManager.unlock_streaming()
        print("Timer completed!")

    def tts_streaming(self):
        """
        Streaming the descriptions by tts service from OpenAI. Check it out at speech.py
        """
        try:
            caption = None
            if self.stop_streaming_reason == "turn":
                self.stop_streaming_reason = ""
                if self.current_caption and not self.ttsManager.stream_locked:
                    self.ttsManager.stop_streaming()
                    self.firebaseWriteManager.send_stop_streaming("turn")
                    self.current_caption = None
                    # self.ttsManager.unlock_streaming()
                    self.start_timer(1.0, self.on_timeout)
                return
                
            
            elif self.stop_streaming_reason == "invoke_agent":
                self.ttsManager.stop_streaming()
                self.stop_streaming_reason = "waiting_for_user_query_and_answer"
                return
            
            elif self.stop_streaming_reason == "waiting_for_user_query_and_answer":
                if self.user_preference.agent_response:
                    caption = self.user_preference.agent_response
                    self.stop_streaming_reason = ""
                    self.user_preference.agent_response = None
                    self.ttsManager.unlock_streaming()
            # print(f"is_streaming: {self.ttsManager.is_streaming}, stream_locked {self.ttsManager.stream_locked}")
            if not self.ttsManager.is_streaming and not self.ttsManager.stream_locked:
                caption = self.get_caption_for_tts() if caption is None else caption
                if caption == None: 
                    self.ttsManager.is_streaming = False
                    return
                
                self.current_caption =  caption
                self.ttsManager.is_streaming = True
                self.firebaseWriteManager.update_caption(caption)
                return

        except queue.Empty:
            pass

    
    def stop_all_tts_threads(self):
        # Signal all threads to stop
        self.stop_threads_event.set()

        # Join all threads
        for thread in self.live_tts_threads:
            if thread.is_alive():
                thread.join()

        # Clear the thread list
        self.live_tts_threads.clear()
        self.stop_threads_event.clear()
        
if __name__ == '__main__':
    import torch
    print(torch.backends.mps.is_available())
