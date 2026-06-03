import os
import base64
import time
from io import BytesIO
import requests
import cv2

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils.classes import SOUND_CLASSES, VISUAL_CLASSES, MANIPULATION_CLASSES, COCO_CLASSES, CUSTOM_CLASSES
from utils.nlp_process import rank_captions, ends_with_be_verb, get_nouns

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.environ.get("OPENAI_API_KEY", "")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

SIMILARITY_THRESHOLD = 0.6

def encode_image_to_base64(numpy_image):
    _, encoded_image = cv2.imencode('.jpg', numpy_image)
    # Convert to base64
    base64_string = base64.b64encode(encoded_image).decode('utf-8')
    return base64_string


def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def classify_user_request(message):
    # if len(CUSTOM_CLASSES) > 0:
    system_role = f"You are a powerful action classification machine, which can understand user's intent on sounds, their manipulations and visuals, and classify them into certain categories. The first one is user's goal on visuals, which contains 'general' and 'specific'. if the user's input is about their goal on visuals, such as I am looking for a sliver laptop, then you should return goal:specidic;sounds:none;manipulation:none;visuals:color;visual_objects:laptop. Another example is that when user says I want to explore the surroundings, then you should return goal:general;sounds:none;manipulation:none;visuals:none; I will introduce the other categories. First, sounds category are {SOUND_CLASSES}. Sound Manipulations are {MANIPULATION_CLASSES}. Visual Categories are {VISUAL_CLASSES}. If visual category falls into object, you should take a look on object categories {CUSTOM_CLASSES} to see which object class best match to user's need. For the return value, You need to use semicolon ; to separate different categories, and use colon to seperate category name and its value, and use comma if you have multiple value. For instance, if user say 'I want you to pause when someone speaking'. Here you need to return goal:none;sounds:speech;manipulation:pause;visuals:none;visual_objects:none to me. Because 'someone speaking' is most similar to 'speech' in the sounds category, and manipulation is similar to pause. Another example for queries on visual information is when user say 'I am curious on color and shape and dog and weather', you should return goal:none;sounds:none;manipulation:none;visuals:color,shape;visual_objects:dog to me. It is because weather does not fall into any visual category and also dog is object that in the object categories. Another example is when user say 'I want you to say less when someone knocking.' in this case you should return goal:none;sounds:knock;manipulation:talk_less;visuals:none;visual_objects:none to me. Another example is when user say 'I want to no more about blue rectangular items', in this case you should return goal:none;sounds:none;manipulation:none;visuals:color,shape;visual_objects:none to me."

    payload = {
        "model": "gpt-4",
        "messages": [
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": message}
                    ],
        "max_tokens": 800,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']

    return res


def process_response_from_user_query(res):
    # print("hello")
    res = res.lower().strip().replace(' ', '')
    category_values = res.split(';')
    # print("category_values", category_values)
    temp = {}
    for c_v in category_values:
        c , v = c_v.split(':')
        temp[c] = v.split(',')
        if 'none' in temp[c]:
            temp[c].remove('none')
    # print(temp)
    return temp


def prepare_inputs(systemrole, image, user_msg):

    base64_image = encode_image_to_base64(image)

    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4o",
        "messages": [
        {
            "role": "system",
            "content": [
                systemrole
            ]
        }, 
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": user_msg, 
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 800
    }

    return payload

def request_gpt4v(s, firebaseWriteManager, firestoreManager):
    system_role          = s['system_role']
    user_msg             = s['user_msg']
    # image                = base64_to_cv2_image(s['frame'])
    # image       = s['frame']
    image                = s['frame_cv2']
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
    
    start_time = time.time()
    payload = prepare_inputs(system_role, image, user_msg)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']

    caption_list = process_response(res)
    
    
    debug = False
    if debug:
        filename = ' '.join(str(e) for e in caption_list)
        cv2.imwrite('img/'+filename+'.jpg', image)

    temp_list = []
    if temp_list is not None: 
        for item in caption_list:
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
                                "source"              : "gpt4"
                            })

    

    print(f"GPT4v----takes {time.time()-start_time} s-----------------------")

    if len(temp_list) ==0: return

    if firebaseWriteManager: 
        caption_queue = firebaseWriteManager.get_caption_buffer()
        if caption_queue == None or caption_queue == "start" or caption_queue==['s', 't', 'a', 'r', 't']:
            if not firebaseWriteManager.is_ranking:
                firebaseWriteManager.is_ranking = True
                rank_list = rank_captions(image, user_goal, temp_list, adj_preference, user_goal_type)
                print("rank_list",  len(rank_list), type(rank_list))
                firebaseWriteManager.update_caption_buffer(rank_list)
                firestoreManager.gpt_update(rank_list)
                firebaseWriteManager.is_ranking = False
                
        else:
            if caption_queue[0]['state_changed_time'] < state_changed_time:
                # similarity = get_frame_similarity(image, frame_history[caption_queue[0]['frame_id']]['frame'])
                # if similarity > SIMILARITY_THRESHOLD or event == "long_new_scene" or event == "long_empty_scene":
                if not firebaseWriteManager.is_ranking:
                    firebaseWriteManager.is_ranking = True
                    rank_list = rank_captions(image, user_goal, temp_list, adj_preference, user_goal_type)
                    print("rank_list",  len(rank_list), type(rank_list))
                    firebaseWriteManager.update_caption_buffer(rank_list)
                    firestoreManager.gpt_update(rank_list)
                    firebaseWriteManager.is_ranking = False

            else:
                print("*********************This gpt output is late********************")
    print(f"GPT4v----takes {time.time()-start_time} s-----------------------\n")
    print("\n\n\n\n")


    return 


def process_response(description, BLIP=False, object_preference=None):
    # print(description)
    description_list = description.strip().split('.')
    # filter out empty string and caption that has less informativeness score 5
    
    if BLIP:
        description_list = [item.strip() for item in description_list if len(item) !=0
                            and cal_caption_informativeness(item,object_preference)>=0] 
    else:
        description_list = [item.strip() for item in description_list if len(item) !=0 
                        and cal_caption_informativeness(item,object_preference)>=0
                        ]
    

    # print("description_list", description_list)
    # filter by pipeline
    # description_list = [filter_pipeline(item) for item in description_list]
    from typing import List
    description_list: List[str]
    # print(description_list)
    output = []
    for i,item in enumerate(description_list):
        item = item.strip().lower()
        item = item.removeprefix('there appears to be')
        item = item.removeprefix("there are")
        item = item.removeprefix('there is')
        item = item.removeprefix("there's")
        item = item.removeprefix("there're")
        item = item.removeprefix("these include")
        item = item.removeprefix("a glimpse of")
        item = item.removeprefix("it displays")
        item = item.removeprefix("there seems to be a part of")
        item = item.removeprefix("there seems to be")
        item = item.removeprefix("this includes")
        item = item.replace(" also", "")
        item = item.replace("a partial glimpse of", "")
        item = item.replace("the image shows", "")
        item = item.replace("the image features", "")
        item = item.replace("the image depicts", "")
        item = item.replace("a blurry photo of", "")
        item = item.replace("a blurry image of", "")
        item = item.replace("a blurry picture of", "")
        item = item.replace("is in the background", "")
        item = item.replace("is in the foreground", "")
        item = item.replace("is positioned in the center of the image", "")
        item = item.replace("is in the center of the image", "")
        item = item.replace("in the center of the image", "")
        item = item.replace("is the central focus of this image", "")
        item = item.replace("are the central focus of this image", "")
        item = item.replace("the central focus of this image", "")
        item = item.replace("is in the image", "")
        item = item.replace("in the image,", "")
        item = item.replace("in the image", "")
        item = item.replace("are in the background", "")
        item = item.replace("in the background", "")
        item = item.replace("are in the foreground", "")
        item = item.replace("in the foreground", "")
        item = item.replace("is depicted in this image", "")
        item = item.replace("is visible", "")
        item = item.replace("are visible", "")
        item = item.replace("it leads to", "")
        item = item.replace("you see", "")
        item = item.replace("visible", "")
        item = item.replace("partially", "")
        item = item.replace("immediate", "")
        item = item.replace("view", "")
        item = item.replace("seen", "")

        if ends_with_be_verb(item): item = get_nouns(item)
        if is_integer(item): continue
        
            
        output.append(item.strip())
        print(f"[{i}] {item}")
    # print(output)
    return output

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def cal_caption_informativeness(caption,object_preference=None):
    # Basic stopwords list from NLTK
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(caption)
    score = 0
    informative_words = [word for word in words if word.lower() not in stop_words]
    score += len(informative_words)
    
    # Deduct points for phrases indicating lack of information
    object_constraint = []
    if object_preference: object_constraint = [item for item in object_preference if object_preference[item] == 'false']
    negative_indicators = ['clockwise', 'ceiling', 'carpet', 'unable', 'request', 'sorry', 'orientation', 'please', 'obscures', 'rotate', 'indistinct', ' no ', ' not ' ,'blurry', 'noticeable', 'hard', 'blurred', 'not possible', 'impossible', 'obscured', 'difficult', 'blurriness', 'indiscernible', 'lack of focus', 'perhaps', 'miscellaneous', 'possibly', 'uncertain', 'unidentified', 'unclear', 'no object', 'due to the perspective']
    # negative_indicators = negative_indicators + object_constraint
    caption = caption.lower()
    for negation in negative_indicators:
        if negation in caption:
            score -= 30  # Deduct 5 points for each negative indicator
    
    # print("object_constraint", object_constraint)
    for object in object_constraint:
        if object in get_nouns(caption):
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            score-=30
    return score


if __name__ == '__main__':
    import time

    # test = ['a person', 'the large painting', 'a smaller painting', 'a black trash bin', 'a metal barrier']
    # print(get_sentence_similarity_spacy(["Describe paintings or poster in detail"], test))
    # print(get_sentence_similarity_spacy(["Describe any person"], test))
    
    image = cv2.imread('/Users/rueichechang/Downloads/test.png')
    systemrole = "You are a helpful visual describer, who can see and describe for blind or visually imparied people. You will not mention this is an image, just describe it, and also don't mention camera issues like blurry or motion. Instead, describe things as you see as in real world but not an image. And you should describe each object with ONLY ONE sentence in maximum. Don't use 'it' to describe an object. DON'T DESCRIBE THINGS THAT ARE NOT IN THE IMAGE. Each sentence should start with a subject. Most importantly, each sentence should be at least 10 words."
    SENTENCE_LENGTH_10 = "at least 10 words"
    SENTENCE_LENGTH_20 = "at least 30 words"
    SENTENCE_LENGTH_5 = "no longer than 5 words"
    sentence_requirement = SENTENCE_LENGTH_20
    user_msg = f"I want to understand each individual object and their visual features. And you should describe each object with ONLY ONE sentence in maximum. Don't use 'it' to describe an object. DON'T DESCRIBE THINGS THAT ARE NOT IN THE IMAGE. Each sentence should be {sentence_requirement}."
    payload = prepare_inputs(systemrole, image, user_msg)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']

    caption_list = process_response(res)
    print(caption_list)
    print('----------------------------------------------')

    temp_list = []
    if temp_list is not None: 
        for item in caption_list:
            temp_list.append({"caption"               :item, 
                                "similarity_score"    : None,
                                "depth_score"         : None,
                            })
            
    test = rank_captions(image, "Find a beige backpack on a dining chair.", temp_list, "specific")
    print(test)