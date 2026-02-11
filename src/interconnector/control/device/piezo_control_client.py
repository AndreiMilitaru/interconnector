"""
author: Andrei Militaru
date: 8th December 2024
"""
import time
import socket
import json


class ControllerClient(object):
    HOST = '10.21.217.17'  
    PORT = 65433        # The port used by the server
    sock = None
    
    def __del__(self):
        self.sock.sendall(b'SHUT_DOWN')
        self.sock.close()

    def __init__(self, HOST=None, PORT=None):
        if HOST is not None:
            self.HOST = HOST
        if PORT is not None:
            self.PORT = PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.HOST, self.PORT))

    @staticmethod
    def decode_reception(msg):
        if msg is None:
            print('Connection to server failed.')
            return None
        msg = msg.decode('utf-8')
        tmp = msg.split('DP_START')
        if len(tmp) == 2:
            tmp = tmp[1].split('DP_STOP')
            if len(tmp) == 2:
                return json.loads(tmp[0])
        else:
            print('Received message from server has unexpected format.')
            return None
            
    def get_x_voltage(self):
        self.sock.sendall(b'READ_VALUE_X')
        msg = self.sock.recv(1024)
        dic = self.decode_reception(msg)
        if dic is not None:
            return dic['VALUE']
        else:
            print('Could not get anything.')
            return None
        
    def set_x_voltage(self, new_value, with_return=False):
        self.sock.sendall('SET_VALUE_X::{:.5f}'.format(new_value).encode('ascii'))
        msg = self.sock.recv(1024)
        if with_return:
            dic = self.decode_reception(msg)
            if dic is not None:
                return dic['VALUE']
            else:
                print('Could not get anything back.')
                return None
            
    def get_y_voltage(self):
        self.sock.sendall(b'READ_VALUE_Y')
        msg = self.sock.recv(1024)
        dic = self.decode_reception(msg)
        if dic is not None:
            return dic['VALUE']
        else:
            print('Could not get anything.')
            return None
        
    def set_y_voltage(self, new_value, with_return=False):
        self.sock.sendall('SET_VALUE_Y::{:.5f}'.format(new_value).encode('ascii'))
        msg = self.sock.recv(1024)
        if with_return:
            dic = self.decode_reception(msg)
            if dic is not None:
                return dic['VALUE']
            else:
                print('Could not get anything back.')
                return None
            
    def get_z_voltage(self):
        self.sock.sendall(b'READ_VALUE_Z')
        msg = self.sock.recv(1024)
        dic = self.decode_reception(msg)
        if dic is not None:
            return dic['VALUE']
        else:
            print('Could not get anything.')
            return None
        
    def set_z_voltage(self, new_value, with_return=False):
        self.sock.sendall('SET_VALUE_Z::{:.5f}'.format(new_value).encode('ascii'))
        msg = self.sock.recv(1024)
        if with_return:
            dic = self.decode_reception(msg)
            if dic is not None:
                return dic['VALUE']
            else:
                print('Could not get anything back.')
                return None
