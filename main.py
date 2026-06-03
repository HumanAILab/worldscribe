"""
Main file that should be run on local laptop
"""
import socket
import threading
import json
import queue
import cv2
import firebase_admin
from firebase_admin import credentials

if not firebase_admin._apps:
    cred = credentials.Certificate('soundcaption-a6e7d-firebase-adminsdk-mwgfx-7e8cba13f0.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://soundcaption-a6e7d-default-rtdb.firebaseio.com/'
    })

from utils.data_processor import DataProcessor
from utils.worldscribe_utils import base64_to_cv2_image
from utils.firebase.firebase_manager import FirebaseManager
from config.config_server import SERVER_IP, VISUAL_SERVER_PORT

# TCP protocol setup
recv_packet_size = 65535
packet_tail = "_TAIL".encode('utf-8') # this is the tail data that identifies the end of the packet which is the string _TAIL

def process_server_visual_data(data):
    # decode the data string from smartphone, which is a json
    server_data_dict_data_str = data.decode("utf-8")
    # decode the json into dictionary 
    server_data_dict = json.loads(server_data_dict_data_str)
    # get the image frame in the form of base64string
    frame_base64string = server_data_dict["camera_data"]['rgb_frame']
    # convert the base64 to the cv2 image
    frame = base64_to_cv2_image(frame_base64string)
    server_data_dict["camera_data"]['rgb_frame'] = frame
    server_data_dict['frame_base64'] = frame_base64string

    # This is the gobal dataProcessor, where we only have one in our system
    global dataProcessor
    # process the frame through yolo
    frame = dataProcessor.process_data_by_yolo(frame, server_data_dict)

    # check if the frame exists
    if type(frame) is type(None): return

    # put the frame into a image queue for display the image using function display_image below
    global image_queue
    image_queue.put(("img_name", frame))
    return

def display_image():
    while True:
        try:
            global image_queue
            # get the image from image_queue and display
            img_name, img = image_queue.get(block=True, timeout=.1)  # poll every 0.1 seconds
            cv2.imshow('Image preview', img)
        except queue.Empty:
            ...  # no new image to display
        key = cv2.pollKey()  # non-blocking
        if key & 0xff == ord('q'):
            cv2.destroyAllWindows()
            return
    

def handle_client(client_socket, addr, mode):
    data = b''
    while True:
        # receive data from the client smartphone   
        data += client_socket.recv(recv_packet_size)
        if data[-len(packet_tail):] == packet_tail:
            # check if this is the end of the data stream by identifying _TAIL
            data = data[:-len(packet_tail)]
            process_server_visual_data(data)
            data = b''

            # respond to the smartphone with this msg
            response = "ImageProccessed"
            client_socket.send(response.encode("utf-8"))

def run_server(server_ip = "127.0.0.1", port = 8000, mode="visual"):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind the socket to the host and port
        server.bind((server_ip, port))
        # listen for incoming connections
        server.listen()
        print(f"Listening on {server_ip}:{port}")

        while True:
            # accept a client connection
            client_socket, addr = server.accept()
            print(f"Accepted connection from {addr[0]}:{addr[1]}")
            # start a new thread to handle the client
            thread = threading.Thread(target=handle_client, args=(client_socket, addr, mode,))
            thread.start()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.close()


if __name__ == '__main__':
    image_queue = queue.Queue() # for display the frames for debugging purposes
    dataProcessor = DataProcessor() # main data processor
    firebaseManager = FirebaseManager(dataProcessor, image_queue) # the gobal firebasemanager
    visualServer = threading.Thread(target=run_server, args=(SERVER_IP, VISUAL_SERVER_PORT,)) # open a thread for server in order to make the display_image() runnable below
    visualServer.start()
    display_image()