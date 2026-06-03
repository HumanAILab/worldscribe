"""firestore manage class"""
import time
import uuid
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import db

firestore_db = firestore.client()

class FirestoreManager:
    def __init__(self):
        self.name = str(time.time())
        
        self.user_ref = firestore_db.collection('users').document(self.name)
        
        self.dataKeys = ['caption',
                         'frame_id',
                         'event',
                         'state_changed_time',
                         'complete_time',
                         'uuid',
                         'user_degree', # when user uploaded the image
                         'frame_base64',
                         'payload',
                         "ids",
                         "system_role",
                         "user_msg",
                         "user_goal",
                         "sentence_requirement",
                         "clss",
                         "similarity_score",
                         "depth_score",
                         ]
    
    def set_name(self, name):
        self.name = name
        self.user_ref = firestore_db.collection('users').document(name)
        return

    def gpt_update(self,caption_list):
        gpt_refs = self.user_ref.collection('gpt_log').document()
        gpt_refs.set({'content':caption_list})
        return
    
    def moondream_update(self,caption_list):
        moondream_refs= self.user_ref.collection('moondream_log').document()
        moondream_refs.set({'content':caption_list})
        return
    
    # def blip2_update(self,caption_list):
    #     blip2_refs= self.user_ref.collection('blip2_log').document()
    #     blip2_refs.set({'content':caption_list})
    #     return
    
    def read_caption_update(self,caption_list):
        read_caption_refs= self.user_ref.collection('read_caption_log').document()
        read_caption_refs.set({'content':caption_list})
        return

    def read_gpt_log(self, collection_name):
        for doc in self.user_ref.collection(collection_name).stream():
            print(f'{doc.id} => {doc.to_dict()}')
        return

if __name__ == '__main__':
    firestoreManager = FirestoreManager()
    