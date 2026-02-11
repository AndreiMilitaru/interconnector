"""
author: Andrei Militaru
date: 8th December 2024
"""

from mdt69x import Controller
import numpy as np
import socket
import json
import sys
from threading import Thread, Lock

class ConnectionHandler(Thread):
    def __init__(self, conn, addr, lock, controller):
        Thread.__init__(self)
        self.conn = conn
        self.addr = addr
        self.lock = lock
        self.controller = controller

    def send_msg(self, dic):
        msg = b'DP_START' + json.dumps(dic).encode('utf-8') + b'DP_STOP'
        self.conn.sendall(msg)

    def run(self):
        while True:
            print('Connected by', self.addr)
            while True:
                data_full = self.conn.recv(1024)
                decoded = data_full.decode('utf-8')
                if not data_full:
                    break
                if decoded == 'READ_VALUE_X':
                    self.lock.acquire()
                    tmp = self.controller.get_x_voltage()
                    self.lock.release()
                    tmp = {'VALUE': tmp}
                    self.send_msg(tmp)
                elif 'SET_VALUE_X' in decoded:
                    new_value = float(decoded.split('::')[-1])
                    self.lock.acquire()
                    self.controller.set_x_voltage(new_value)
                    self.lock.release()
                    tmp = {'VALUE': 0}
                    self.send_msg(tmp)
                elif 'READ_VALUE_Y' in decoded: 
                    self.lock.acquire()
                    tmp = self.controller.get_y_voltage()
                    self.lock.release()
                    tmp = {'VALUE': tmp}
                    self.send_msg(tmp)
                elif 'SET_VALUE_Y' in decoded:
                    new_value = float(decoded.split('::')[-1])
                    self.lock.acquire()
                    self.controller.set_y_voltage(new_value)
                    self.lock.release()
                    tmp = {'VALUE': 0}
                    self.send_msg(tmp)
                elif 'READ_VALUE_Z' in decoded: 
                    self.lock.acquire()
                    tmp = self.controller.get_z_voltage()
                    self.lock.release()
                    tmp = {'VALUE': tmp}
                    self.send_msg(tmp)
                elif 'SET_VALUE_Z' in decoded:
                    new_value = float(decoded.split('::')[-1])
                    self.lock.acquire()
                    self.controller.set_z_voltage(new_value)
                    self.lock.release()
                    tmp = {'VALUE': 0}
                    self.send_msg(tmp)
                else:
                    self.conn.sendall(b'Did not understand request.')


class ControllerServer(object):

    HOST = '10.21.217.17'
    PORT = 65433
    sock = None

    def __del__(self):
        self.sock.close()
    
    def __init__(self, N_listens=20):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen(N_listens)
        self.lock = Lock()
        self.controller = Controller("COM3")
        self.clients = []
        self.handlers = []

    def start_server(self):
        print('Starting server.')
        while True:
            conn, addr = self.sock.accept()
            self.clients.append((conn, addr))
            handler = ConnectionHandler(conn, addr, self.lock, self.controller)
            self.handlers.append(handler)
            handler.start()

if __name__ == '__main__':
    server = ControllerServer()
    server.start_server()