
import traceback
import json
from collections import deque
import ast
import time
import base64
import os
import torch
import requests
import threading
import zlib
import json
import base64
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
import queue
import uuid
from datetime import datetime
from utils.gpt4v import process_response, prepare_inputs, headers

from utils.worldscribe_utils import base64_to_cv2_image, encode_image_to_base64
from utils.firebase.firebase_manager import FirebaseWriteManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from utils.nlp_process import rank_captions, preference_check

# Initialize the app with a service account
if not firebase_admin._apps:
    cred = credentials.Certificate('soundcaption-a6e7d-firebase-adminsdk-mwgfx-7e8cba13f0.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://soundcaption-a6e7d-default-rtdb.firebaseio.com/'
    })
from utils.firebase.firestore_manager import FirestoreManager

torch.cuda.empty_cache()
moondream_device = "cuda:1" # TODO: change 1 and 0

device1 = torch.device(moondream_device if torch.cuda.is_available() else "cpu")

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision, attn_implementation="flash_attention_2"
).to(device1, torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

class RemoteServer:

    def __init__(self):
        self.firestoreManager = FirestoreManager()

        self.ref_image_data_start = True
        self.ref_image_data = db.reference('image_data/')
        self.ref_image_data.listen(self.listener_image_data)
        
        self.ref_moondream = db.reference('moondream_buffer/')
        
        self.ref_user_timestamp = db.reference('user_timestamp/')
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ref_user_timestamp.set(self.timestamp)
        
        self.ref_user_goal = db.reference('user_query/input')
        self.ref_user_goal.listen(self.listener_user_goal)
        
        self.user_goal = db.reference('user_query/input').get()
        self.firebaseWriteManager = FirebaseWriteManager()
        print(f"{'*'*30}\n")
        print(f"USER GOAL IS: {self.user_goal}")
        print(f"{'*'*30}\n")
        print(f"REMOTE SERVER STARTS TO RECEIVING TASKS...")
        print(f"{'*'*30}\n")

        self.ref_adj_start = True
        self.ref_adjective = db.reference('adjective_categories/')
        self.adj_preference = self.ref_adjective.get()
        self.ref_adjective.listen(self.lister_adj_data)
    
        self.ref_user_name_start = True
        self.ref_user_name = db.reference('user_name/')
        self.ref_user_name.listen(self.listen_user_name)

        self.object_preference = {}
        self.ref_object_start = True
        self.ref_object = db.reference('object_categories/')
        self.ref_object.listen(self.listen_object_data)
    
        self.moondream_is_working = False
        self.folder_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.threads = []  # Track running threads
        self.stop_threads = False  # Flag to stop all running threads
        
        self.latest_frame_id = None
        

    def listen_object_data(self, event):
        try:
            if self.ref_object_start:
                self.ref_object_start = False
            else:
                self.object_preference = event.data
                print(event.data)

        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

        return

    def listen_user_name(self,event):
        
        try:
            if self.ref_user_name_start:
                self.ref_user_name_start = False
            else:
                print(event.data)
                user_name = str(event.data)
                self.firestoreManager.set_name(user_name)
        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback


        return
    
    
    def lister_adj_data(self, event):
        try:
            if self.ref_adj_start:
                self.ref_adj_start = False
            else:
                print(event)
                adj = str(event.path).split('/')[1]
                self.adj_preference[adj] = event.data
                print(self.adj_preference)

        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

        return
    
    def listener_user_goal(self, event):
        try:
            # if self.ref_image_data_start:
            #     self.ref_image_data_start = False
            # else:
            
            s = event.data
            self.user_goal = str(s)
            print(s)
        except Exception as e:
            print("Errors: ", e)
    
    def stop_all_threads(self):
        """Sets the flag to stop all running threads and waits for them to exit."""
        self.stop_threads = True
        for t in self.threads:
            if t.is_alive():
                t.join()  # Wait for the thread to finish
        
        # Reset threads list after stopping all
        self.threads.clear()
        self.stop_threads = False
        
    def listener_image_data(self, event):
        try:
            if self.ref_image_data_start:
                self.ref_image_data_start = False
            
            else:
                # s = self.decompress_data(event.data)
                s = dict(event.data)
                s['adj_preference'] = self.adj_preference
                s['frame_cv2'] = base64_to_cv2_image(s['frame'])
                frame_id = s["frame_id"]

                # Update the latest frame ID before adding to queue
                self.latest_frame_id = frame_id
                                
                if 'ids' not in s.keys(): s['ids'] = []
                if 'clss' not in s.keys(): s['clss'] = []
                
                self.stop_all_threads()

                # Create new threads
                t1 = threading.Thread(target=self.request_gpt4v, args=(s, self.firebaseWriteManager, self.firestoreManager))
                t2 = threading.Thread(target=self.moondream_inference, args=(s,))

                # Start and track threads
                t1.start()
                t2.start()
                self.threads.extend([t1, t2])  
                
        except Exception as e:
            print("Errors: ", e)
            traceback.print_exc()  # This will print the full traceback

    def get_moondream_caption_buffer(self):
        values = self.ref_moondream.get()
        if values == "start" or values == None: return "start"
        values = list(self.ref_moondream.get())
        return values
    
    def logging(self, source, image, temp_list):
        output_dict = {
            "list": temp_list,
            "image": encode_image_to_base64(image)
        }

        # Create the folder structure
        subfolder = source + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join("ASSETS2025", self.timestamp, subfolder)
        os.makedirs(folder, exist_ok=True)  # Ensure folder exists

        # Save the image
        image_address = os.path.join(folder, f"{subfolder}.jpg")
        output_dict['image_address'] = image_address
        output_dict['json_address']  = os.path.join(folder, f"{subfolder}.json")
        output_dict['filename']      = subfolder
        cv2.imwrite(image_address, image)  # Save the image file

        # Save the JSON log
        json_address = os.path.join(folder, f"{subfolder}.json")
        with open(json_address, "w", encoding="utf-8") as json_file:
            json.dump(output_dict, json_file, ensure_ascii=False, indent=4)

        print(f"Saved image: {image_address}")
        print(f"Saved JSON log: {json_address}")
        
        
    
    
    
    def request_gpt4v(self, s, firebaseWriteManager, firestoreManager):
        system_role          = s['system_role']
        user_msg             = s['user_msg']
        # image                = base64_to_cv2_image(s['frame'])
        # image       = s['frame']
        image                = s['frame_cv2']
        image                = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        event                = s['event']
        frame_id             = s['frame_id']
        uuid                 = s['uuid']
        user_goal            = s['user_goal']
        state_changed_time   = s['state_changed_time']
        user_degree          = s['user_degree']
        ids                  = s['ids']
        system_role          = s['system_role']
        user_msg             = s['user_msg']
        user_goal            = s['user_goal']
        user_goal_type       = s['user_goal_type']
        sentence_requirement = s['sentence_requirement']
        clss                 = s['clss']
        adj_preference       = s['adj_preference']
        
        if frame_id != self.latest_frame_id:
            print(f"GPT Skipping outdated frame {frame_id}, latest is {self.latest_frame_id}")
            return
        
        start_time = time.time()
        user_msg = "Describe objects with their color and relative positions."
        payload = prepare_inputs(system_role, image, user_msg)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        res = response.json()['choices'][0]['message']['content']
        
        if frame_id != self.latest_frame_id:
            print(f"GPT Skipping outdated frame {frame_id}, latest is {self.latest_frame_id}")
            return

        caption_list = process_response(res)
        
        debug = False
        if debug:
            filename = ' '.join(str(e) for e in caption_list)
            cv2.imwrite('img/'+filename+'.jpg', image)

        temp_list = []
        if temp_list is not None: 
            for item in caption_list:
                if not item: continue
                temp_list.append({"caption"               :item, 
                                    "frame_id"            : frame_id, 
                                    "event"               : event, 
                                    "state_changed_time"  : state_changed_time,
                                    "complete_time"       : time.time(),
                                    "uuid"                : uuid,
                                    "user_degree"         : user_degree,
                                    "payload"             : payload['messages'][1]['content'][0]['text'],
                                    # "frame_base64"        : s['frame'],
                                    "ids"                 : ids,
                                    "system_role"         : system_role,
                                    "user_msg"            : user_msg,
                                    "user_goal"           : user_goal,
                                    "user_goal_type"      : user_goal_type,
                                    "sentence_requirement": sentence_requirement,
                                    "clss"                : clss,
                                    "similarity_score"    : None,
                                    "depth_score"         : None,
                                    "adj_preference"      : adj_preference,
                                    "source"              : "gpt4",
                                    "timestamp"           : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "timestamp_s"         : time.time()
                                })

        

        print(f"GPT4v----takes {time.time()-start_time} s-----------------------")

        if len(temp_list) ==0: return

        rank_list = rank_captions(image, user_goal, temp_list, adj_preference, user_goal_type)
        # print("rank_list",  len(rank_list), type(rank_list))
        
        if frame_id != self.latest_frame_id:
            print(f"GPT Skipping outdated frame {frame_id}, latest is {self.latest_frame_id}")
            return
        
        firebaseWriteManager.update_caption_buffer(rank_list)
        firestoreManager.gpt_update(rank_list)
        # self.log_memory(packet=s)
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.logging("gpt", image, temp_list)
        
        
        # if firebaseWriteManager: 
        #     caption_queue = firebaseWriteManager.get_caption_buffer()
        #     if caption_queue == None or caption_queue == "start" or caption_queue==['s', 't', 'a', 'r', 't']:
        #         if not firebaseWriteManager.is_ranking:
        #             firebaseWriteManager.is_ranking = True
        #             rank_list = rank_captions(image, user_goal, temp_list, adj_preference, user_goal_type)
        #             print("rank_list",  len(rank_list), type(rank_list))
        #             firebaseWriteManager.update_caption_buffer(rank_list)
        #             firestoreManager.gpt_update(rank_list)
        #             firebaseWriteManager.is_ranking = False
                    
        #     else:
        #         if caption_queue[0]['state_changed_time'] < state_changed_time:
        #             # similarity = get_frame_similarity(image, frame_history[caption_queue[0]['frame_id']]['frame'])
        #             # if similarity > SIMILARITY_THRESHOLD or event == "long_new_scene" or event == "long_empty_scene":
        #             if not firebaseWriteManager.is_ranking:
        #                 firebaseWriteManager.is_ranking = True
        #                 rank_list = rank_captions(image, user_goal, temp_list, adj_preference, user_goal_type)
        #                 print("rank_list",  len(rank_list), type(rank_list))
        #                 firebaseWriteManager.update_caption_buffer(rank_list)
        #                 firestoreManager.gpt_update(rank_list)
        #                 firebaseWriteManager.is_ranking = False

        #         else:
        #             print("*********************This gpt output is late********************")
        print(f"GPT4v----takes {time.time()-start_time} s-----------------------\n")
        print("\n\n\n\n")


        return 
    
    
    def log_memory(self, packet):
        frame                 = packet['frame_cv2']
        packet['timestamp']   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        packet['timestamp_s'] = time.time()
        folder             = os.path.join('worldscribe_assets_data', self.folder_path, packet['timestamp'])
        # "memory_data/" + str(spoken_timestamp)
        
        print(f"[MEMORY LOGGING] Saving to {folder}")
    
        os.makedirs(folder, exist_ok=True)
        
        
        full_image = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        full_image_path = os.path.join(folder, "full_image.png")
        cv2.imwrite(full_image_path, full_image)
        
        json_path = os.path.join(folder, "metadata.json")            
        
        # Save metadata
        
        packet['full_image_path'] = full_image_path
        packet['json_data_path'] = json_path

        
        # self.memoryManager.add_data_to_table("2d_object", packet_metadata)
        with open(json_path, "w") as json_file:
            json.dump(packet, json_file, indent=4)
        
        
    
    def moondream_inference(self, s, prompt="Describe the image shortly with object color and their relative positions."):
        start = time.time()

        # image_np = base64_to_cv2_image(s['frame'])
        image_np = s['frame_cv2']
        frame_id = s["frame_id"] 
        image_np = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 
        # image_np = s['frame']
        image = Image.fromarray(image_np_bgr)
        
        if frame_id != self.latest_frame_id:
            print(f"Skipping outdated frame {frame_id}, latest is {self.latest_frame_id}")
            return
        
        #####
        # start_time = time.time()
        # system_role = "You are a visual describer that is able to provide a general overview of the objects and their color in the image within one sentence."
        # user_msg = "Can you describe the objects and their colors in general?"
        # image = image_np
        # payload = prepare_inputs(system_role, image, user_msg)
        # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # generated_text = response.json()['choices'][0]['message']['content']
        # # print(f"moondream-GPT4v----takes {time.time()-start_time} s-----------------------")
        #####
        
        enc_image = model.encode_image(image).to(device1)
        generated_text = model.answer_question(enc_image, prompt, tokenizer)
        self.moondream_is_working = False      
        
        if frame_id != self.latest_frame_id:
            print(f"Skipping outdated frame {frame_id}, latest is {self.latest_frame_id}")
            return
        
        print("***********************************")
        print("*****  Moondream is working  ******")
        print("***********************************")

        # moondream_caption_queue = self.get_moondream_caption_buffer()
        frame_id             = s['frame_id']
        event                = s['event']
        uuid                 = s['uuid']
        state_changed_time   = s['state_changed_time']
        user_degree          = s['user_degree']
        ids                  = s['ids']
        system_role          = s['system_role']
        user_msg             = s['user_msg']
        user_goal            = s['user_goal']
        user_goal_type       = s['user_goal_type'] 
        sentence_requirement = s['sentence_requirement']
        clss                 = s['clss']

        # caption_list = [generated_text]
        
        print("Original Text:", generated_text)
        print("Original Text:", generated_text)
        print("Original Text:", generated_text)
        print("Original Text:", generated_text)
        caption_list = process_response(generated_text, object_preference=self.object_preference)
        caption_list = [". ".join(caption_list)]
        print("Generated Text:", caption_list)
        print("Generated Text:", caption_list)
        print("Generated Text:", caption_list)
        print("Generated Text:", caption_list)
        temp_list = []
        if temp_list is not None: 
            for item in caption_list:
                if not item: continue
                temp_list.append({"caption"               : item, 
                                    "frame_id"            : frame_id, 
                                    "event"               : event, 
                                    "state_changed_time"  : state_changed_time,
                                    "complete_time"       : time.time(),
                                    "uuid"                : uuid,
                                    "user_degree"         : user_degree,
                                    "payload"             : prompt,
                                    # "frame_base64"        : s['frame'],
                                    "ids"                 : ids,
                                    "system_role"         : system_role,
                                    "user_msg"            : user_msg,
                                    "user_goal"           : user_goal,
                                    "user_goal_type"      : user_goal_type,
                                    "sentence_requirement": sentence_requirement,
                                    "clss"                : clss,
                                    "similarity_score"    : None,
                                    "depth_score"         : None,
                                    "adj_preference"      : self.adj_preference ,
                                    "source"              : "moon",
                                    "timestamp"           : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "timestamp_s"         : time.time()
                                })
        
        self.ref_moondream.set(temp_list)
        self.firestoreManager.moondream_update(temp_list)
        self.logging("moon", image_np, temp_list)
        # self.log_memory(packet=s)
                

        print(f"Moondream--for frame {s['frame_id']}---{s['event']}-takes {time.time()-start} s-----------------------\n")
        print("\n\n\n\n")
        return generated_text
    
if __name__ == '__main__':
    remote_server = RemoteServer()
