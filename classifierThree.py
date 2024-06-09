import argparse
import csv
import time
import numpy as np
import joblib
import modelP

import pythonosc.dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

#model_dict = joblib.load('./trained_model.p')
data_buffer = []

def print_petal_stream_handler(unused_addr, *args):
    sample_id = args[0]
    unix_ts = args[1] + args[2]
    lsl_ts = args[3] + args[4]
    data = args[5:]
    print(
        f'sample_id: {sample_id}, unix_ts: {unix_ts}, '
        f'lsl_ts: {lsl_ts}, data: {data}'
    )
    data_buffer.append((sample_id, unix_ts, lsl_ts, data))

def save_data_every_2_seconds():
    global data_buffer
    while True:
        time.sleep(0.5)  # Wait for 2 seconds
        # Save data to a variable
        saved_data = data_buffer
        print(model.predict(saved_data))
        data_buffer = []  # Clear the buffer

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ip', type=str, required=False,
                    default="127.0.0.1", help="The ip to listen on")
parser.add_argument('-p', '--udp_port', type=str, required=False, default=14739,
                    help="The UDP port to listen on")
parser.add_argument('-t', '--topic', type=str, required=False,
                    default='/PetalStream/eeg', help="The topic to print")
args = parser.parse_args()

dispatcher = pythonosc.dispatcher.Dispatcher()
dispatcher.map(args.topic, print_petal_stream_handler)

server = pythonosc.osc_server.ThreadingOSCUDPServer(
    (args.ip, args.udp_port),
    dispatcher
)

print("Serving on {}".format(server.server_address))
server.serve_forever()