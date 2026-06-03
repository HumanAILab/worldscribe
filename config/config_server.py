"""
config file to store the ip and ports
Note: this is just an example, do not push your actual IP onto github for security reasons
"""
import os

SERVER_IP = os.environ.get('SERVER_IP', '127.0.0.1') # TODO: place your IP here
VISUAL_SERVER_PORT = 8080 # port number
