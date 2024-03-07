import socket, os

class CommUnit:

    def __init__(self, split_number, number_of_client = 4, port=25258, client='0.0.0.0'):
        self.split_number = split_number
        self.number_of_client = number_of_client

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((client, port))
        self.server.listen()

        self.client = []
        self.addr = []
        self.client_path = []
        self.throughput_list = []
        self.num_images_list = []
        self.SSD_numbers = []

        for i in range(self.number_of_client):
            client_tmp, addr_tmp = self.server.accept()
            self.client.append(client_tmp)
            self.addr.append(addr_tmp)

        self.send_message(str(split_number))
    
    def get_SSD_path(self):
        for i in range(self.number_of_client):
            while True:
                data = self.client[i].recv(4096)
                data = data.decode()
                if data[:4] == 'dir:':
                    print(f"The path of SSD client:{self.addr[i][0]}, {data[4:]}")
                    self.client_path.append(data[4:])
                    break

    def get_message(self, msg):
        for i in range(self.number_of_client):
            while True:
                data = self.client[i].recv(4096)
                data = data.decode()
                if data == msg:
                    print("Received from", self.addr[i], data)
                    break
        for i in range(self.number_of_client):
            self.client[i].sendall(data.encode())
	
    def send_message(self, msg, num_init=False):
        for i in range(self.number_of_client):
            msg += f' {i}'*num_init
            self.client[i].sendall(msg.encode())
    
    def load_balancing(self):
        ### Get the throughput list from SOFA SSDs
        for i in range(self.number_of_client):
            while True:
                data = self.client[i].recv(4096)
                data = data.decode()
                if data != "":
                    split_data = data.split()
                    self.throughput_list.append(float(split_data[0]))
                    self.num_images_list.append(int(split_data[1]))
                    self.feature_dim = int(split_data[2])
                    self.SSD_numbers.append(int(split_data[3]))
                    print(f'Received from {self.addr[i]}, Throughput: {split_data[0]}, Images: {split_data[1]}')
                    break
        self.SSD_numbers.sort()
        ### Perform load balancing
        print("SSD Load Balancing...", self.throughput_list)
        t_total = sum(self.throughput_list)
        n_total = sum(self.num_images_list)
        prop_list = []
        load_list = []
        balance_list = []
        move_list = []
        for i in range(self.number_of_client):
            prop_list.append(self.throughput_list[i]/t_total)
            load_list.append(int(prop_list[i]*n_total))
            balance_list.append(self.num_images_list[i])
            c_info_list = [0,0]
            move_list.append(c_info_list)
        if self.load_balance:
            for j in range(10):
                for i in range(self.number_of_client):
                    if load_list[i] <= balance_list[i]:
                        continue
                    elif load_list[i] > balance_list[i]:
                        diff = load_list[i]-balance_list[i]
                        balance_list[(i+1) % self.number_of_client] -= diff
                        move_list[i][1] += diff
                        move_list[(i+1) % self.number_of_client][0] += diff
                        balance_list[i] = load_list[i]
                if load_list == balance_list:
                    break
            print('Load-Balancing Done!!!')
        for i in range(self.number_of_client):
            dec = str(move_list[i][0])
            ext = str(move_list[i][1])
            #data set path must be modified
            extra_path = "/home/SOFA/SSD/merge/shuffle/"+str(((i+1) % len(self.client)))+"_train_image_shuffle.dat"
            extra_label_path = "/home/SOFA/SSD/merge/shuffle/"+str(((i+1) % len(self.client)))+"_train_label_shuffle.dat"
            send_msg = dec+" "+ext+" "+extra_path+" "+extra_label_path
            self.client[i].sendall(send_msg.encode())

    def close_client(self):
        for i in range(self.number_of_client):
            self.client[i].close()
        self.server.close()
