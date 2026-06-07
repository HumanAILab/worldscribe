
import traceback
import json
from collections import deque
import ast
import time
import base64
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

from utils.gpt4v import process_response, request_gpt4v

from utils.worldscribe_utils import base64_to_cv2_image, encode_image_to_base64
from utils.firebase.firebase_manager import FirebaseWriteManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from utils.nlp_process import rank_captions, preference_check

# Initialize the app with a service account
from utils.env_config import FIREBASE_CREDENTIALS, FIREBASE_DATABASE_URL
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DATABASE_URL
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

        self.threads = []  # Track running threads
        self.stop_threads = False  # Flag to stop all running threads

    def decompress_data(self, data):
        """Decompress data using zlib."""
        compressed_bytes = base64.b64decode(data)
        decompressed_data = zlib.decompress(compressed_bytes).decode("utf-8")
        return json.loads(decompressed_data)

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
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                
                if 'ids' not in s.keys(): s['ids'] = []
                if 'clss' not in s.keys(): s['clss'] = []
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                # s['frame'] = base64_to_cv2_image(s['frame'])
                # threading.Thread(target=request_gpt4v, args=(s,
                #                                              self.firebaseWriteManager,
                #                                              self.firestoreManager
                #                                              )).start()

                # if not self.moondream_is_working: 
                #     self.moondream_is_working = True
                #     self.moondream_inference(s)
                
                self.stop_all_threads()

                # Create new threads
                t1 = threading.Thread(target=request_gpt4v, args=(s, self.firebaseWriteManager, self.firestoreManager))
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
    
    def moondream_inference(self, s, prompt="provide general description of the image"):
        start = time.time()
        is_general = False
        if prompt is not None:
            is_general = True
        else:
            prompt = "Describe each individual object in short sentences."
        
        # image_np = base64_to_cv2_image(s['frame'])
        image_np = s['frame_cv2']
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 
        # image_np = s['frame']
        image = Image.fromarray(image_np)
        enc_image = model.encode_image(image).to(device1)
        generated_text = model.answer_question(enc_image, prompt, tokenizer)
        self.moondream_is_working = False      
        
        print("Moondream is working1")
        
        if generated_text and is_general: 
            print("Moondream is working2")

            moondream_caption_queue = self.get_moondream_caption_buffer()
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

            
            title = ""

            caption_list = process_response(generated_text, object_preference=self.object_preference)
            temp_list = []
            if temp_list is not None: 
                for item in caption_list:
                    temp_list.append({"caption"               : title + item, 
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
                                        "source"              : "moon"
                                    })
                    
            if moondream_caption_queue == None or moondream_caption_queue == "start" or moondream_caption_queue ==['s', 't', 'a', 'r', 't']:
                print("Moondream is working3")

                print(f"Moondream----takes {time.time()-start} s----------{generated_text}\n")
                self.ref_moondream.set(temp_list)
                self.firestoreManager.moondream_update(temp_list)
            else:
                print("Moondream is working4")

                if moondream_caption_queue[0]['state_changed_time'] < state_changed_time:
                    print("Moondream is working5")

                    print(f"Moondream----takes {time.time()-start} s-------------{generated_text}\n")
                    self.ref_moondream.set(temp_list)
                    self.firestoreManager.moondream_update(temp_list)

                else:
                    print("*********************This moondream output is late********************")
        
        elif generated_text and not is_general: 
            moondream_caption_queue = self.get_moondream_caption_buffer()
            frame_id           = s['frame_id']
            event              = s['event']
            uuid               = s['uuid']
            state_changed_time = s['state_changed_time']
            user_degree        = s['user_degree']
            ids                  = s['ids']
            system_role          = s['system_role']
            user_msg             = s['user_msg']
            user_goal            = s['user_goal']
            user_goal_type       = s['user_goal_type']
            sentence_requirement = s['sentence_requirement']
            clss                 = s['clss']
            
            title = ""

            caption_list = process_response(generated_text, object_preference=self.object_preference)
            temp_list = []
            if temp_list is not None: 
                for item in caption_list:
                    temp_list.append({"caption"               : title + item, 
                                        "frame_id"            : frame_id, 
                                        "event"               : event, 
                                        "state_changed_time"  : state_changed_time,
                                        "complete_time"       : time.time(),
                                        "uuid"                : uuid,
                                        "user_degree"         : user_degree,
                                        "payload"             : prompt,
                                        "frame_base64"        : s['frame'],
                                        "ids"                 : ids,
                                        "system_role"         : system_role,
                                        "user_msg"            : user_msg,
                                        "user_goal"           : user_goal,
                                        "user_goal_type"      : user_goal_type,
                                        "sentence_requirement": sentence_requirement,
                                        "clss"                : clss,
                                        "similarity_score"    : None,
                                        "depth_score"         : None,
                                        "adj_preference"      : self.adj_preference,
                                        "source"              : "moon"

                                    })
           

            if moondream_caption_queue == None or moondream_caption_queue == "start" or moondream_caption_queue ==['s', 't', 'a', 'r', 't']:
                print(f"Moondream----takes {time.time()-start} s-----------------------\n")
                rank_list = rank_captions(image_np, self.user_goal, temp_list, self.adj_preference, user_goal_type)
                self.ref_moondream.set(rank_list)
                self.firestoreManager.moondream_update(temp_list)

                # print(rank_list)
            else:
                if moondream_caption_queue[0]['state_changed_time'] < state_changed_time:
                    print(f"Moondream----takes {time.time()-start} s-----------------------\n")
                    rank_list = rank_captions(image_np, self.user_goal, temp_list, self.adj_preference, user_goal_type)
                    self.ref_moondream.set(rank_list)
                    self.firestoreManager.moondream_update(temp_list)

                    # print(rank_list)
                else:
                    # rank_list = rank_captions(image_np, self.user_goal, temp_list)
                    # self.ref_moondream.set(rank_list)
                    # print(rank_list)
                    print("*********************This moondream output is late********************")
                    
        print(f"Moondream--for frame {s['frame_id']}---{s['event']}-takes {time.time()-start} s-----------------------\n")
        print("\n\n\n\n")
        return generated_text
    
if __name__ == '__main__':
    remote_server = RemoteServer()
