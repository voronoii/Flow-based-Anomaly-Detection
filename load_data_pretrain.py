
import csv
import re
import numpy as np
from sklearn.preprocessing import minmax_scale
from keras.preprocessing.text import Tokenizer


# (CIDDS-001) Date first seen	/Duration/	Proto/	Src IP Addr	/Src Pt	/Dst IP Addr/	Dst Pt	/Packets/	Bytes	/Flows	/Flags	/Tos	/attackType	attackID	attackDescription
# /Duration/	Proto/	Src IP Addr/ Src Pt	/ /Packets/	Bytes	/Flows




def load_data_4():
    file = open("./UNSW-NB15_1.csv" , 'r', encoding='utf-8')
    rdr = csv.reader(file)


    ### attributes : [ iip dest, proto, dest port, duration ]
    ip_data = []
    proto = []
    dest_port = []
    
    duration = []
    packet = []
    byte = []
    # flow = []

    label = []

    for idx, line in enumerate(rdr):

        ### label ###
        if line[48] == 0:
            label.append(0)
        else:
            label.append(1)

        ### ip address ###

        ip = line[0].strip()
        ip_data.append(ip)


        ### proto ###

        proto.append(line[4])

        ### dest port (devided by maximum port number  65534)
        try:
            dest_port.append(line[3])
        except ValueError:
            continue

        ### duration, pack, byte, flow  >>>  continuous ###
        duration.append(line[6])
        packet.append(line[16])
        byte.append(line[7])
        # flow.append(line[9])
        

    print(ip_data[0], proto[0], dest_port[0], duration[0], packet[0], byte[0])



    file.close()

   

    ### ip, proto, port (categorical) ###

    ip_set = set(ip_data)
    ip_dict = {}
    i = 1
    for ip in ip_set:
        ip_dict[ip] = i
        i+=1

    ip_value = minmax_scale(list(ip_dict.values()), feature_range=(0, 1))
    i = 0
    for ipp in ip_dict.items():
        ip_dict[ipp[0]] = ip_value[i]
        i+=1

    new_ip_data = []
    for ip_ in ip_data:
        new_ip_data.append(ip_dict[ip_])

    del ip_data
    del ip_dict




    proto_data = set(proto)
    proto_dict = {}
    i = 1
    for pro in proto_data:
        proto_dict[pro] = i
        i += 1

    proto_value = minmax_scale(list(proto_dict.values()), feature_range=(0, 1))
    i = 0
    for prot in proto_dict.items():
        proto_dict[prot[0]] = proto_value[i]
        i += 1

    new_proto = []
    for p in proto:
        new_proto.append(proto_dict[p])


    port_data = set(dest_port)
    port_dict = {}
    i = 1
    for port in port_data:
        port_dict[port] = i
        i += 1

    port_value = minmax_scale(list(port_dict.values()), feature_range=(0, 1))
    i = 0
    for portt in port_dict.items():
        port_dict[portt[0]] = port_value[i]
        i += 1

    new_port = []
    for p in dest_port:
        new_port.append(port_dict[p])

        
    
    # duation, packet, byte, flow (continuous)
    
    duration = minmax_scale(duration, feature_range=(0, 1))
    packet = minmax_scale(packet, feature_range=(0, 1))
    byte = minmax_scale(byte, feature_range=(0, 1))
    # flow = minmax_scale(flow, feature_range=(0, 1))
    
    
    print(len(new_ip_data), len(new_proto), len(duration))
    
    dataset = []
    for i in range(len(duration)):
        temp = []
        temp.append(new_ip_data[i])
        temp.append(new_proto[i])
        temp.append(new_port[i])
        
        temp.append(duration[i])
        temp.append(packet[i])
        temp.append(byte[i])
        # temp.append(flow[i])

        dataset.append(temp)


    print(dataset[3])
    dataset = np.array(dataset)

    print(len(dataset), len(label))

    return dataset, label


def load_data_4_dup():
    file = open(
        "/home/selab/mj/GANTest/dataset/CIDDS-001(flow-based)/traffic/OpenStack/CIDDS-001-internal-week3_modified.csv",
        'r')
    rdr = csv.reader(file)

    ### attributes : [ iip dest, proto, dest port, duration ]
    ip_data = []
    proto = []
    dest_port = []

    duration = []
    packet = []
    byte = []
    # flow = []

    for idx, line in enumerate(rdr):

        ### ip address ###

        ip = line[5].strip()
        ip_data.append(ip)

        ### proto ###

        proto.append(line[2])

        ### dest port (devided by maximum port number  65534)
        try:
            dest_port.append(line[6])
        except ValueError:
            continue

        ### duration, pack, byte, flow  >>>  continuous ###
        duration.append(line[1])
        packet.append(line[7])
        byte.append(line[8])
        # flow.append(line[9])

    file.close()

    ### ip, proto, port (categorical) ###

    ip_set = set(ip_data)
    ip_dict = {}
    i = 1
    for ip in ip_set:
        ip_dict[ip] = i
        i += 1

    ip_value = minmax_scale(list(ip_dict.values()), feature_range=(0, 1))
    i = 0
    for ipp in ip_dict.items():
        ip_dict[ipp[0]] = ip_value[i]
        i += 1

    new_ip_data = []
    for ip_ in ip_data:
        new_ip_data.append(ip_dict[ip_])

    del ip_data
    del ip_dict

    proto_data = set(proto)
    proto_dict = {}
    i = 1
    for pro in proto_data:
        proto_dict[pro] = i
        i += 1

    proto_value = minmax_scale(list(proto_dict.values()), feature_range=(0, 1))
    i = 0
    for prot in proto_dict.items():
        proto_dict[prot[0]] = proto_value[i]
        i += 1

    new_proto = []
    for p in proto:
        new_proto.append(proto_dict[p])

    port_data = set(dest_port)
    port_dict = {}
    i = 1
    for port in port_data:
        port_dict[port] = i
        i += 1

    port_value = minmax_scale(list(port_dict.values()), feature_range=(0, 1))
    i = 0
    for portt in port_dict.items():
        port_dict[portt[0]] = port_value[i]
        i += 1

    new_port = []
    for p in dest_port:
        new_port.append(port_dict[p])

    # duation, packet, byte, flow (continuous)

    duration = minmax_scale(duration, feature_range=(0, 1))
    packet = minmax_scale(packet, feature_range=(0, 1))
    byte = minmax_scale(byte, feature_range=(0, 1))
    # flow = minmax_scale(flow, feature_range=(0, 1))

    print(len(new_ip_data), len(new_proto), len(duration))

    dataset = []
    for i in range(len(duration)):
        temp = []
        temp.append(new_ip_data[i])
        temp.append(new_proto[i])
        temp.append(new_port[i])

        temp.append(duration[i])
        temp.append(packet[i])
        temp.append(byte[i])
        # temp.append(flow[i])

        dataset.append(temp)

    print(dataset[3])
    dataset = np.array(dataset)

    return dataset










def prepare_testset_with_label():
    """CIDDS-001 week1.csv / attacker = 850870 / normal = 1048575 """

    file = open("/home/selab/mj/GANTest/dataset/CIDDS-001(flow-based)/traffic/OpenStack/CIDDS-001-internal-week1_modified.csv", 'r')
    rdr = csv.reader(file)

    ### attributes : [ iip dest, proto, dest port, duration ]
    ip_data = []
    proto = []
    dest_port = []
    attrs = []


    for idx, line in enumerate(rdr):

        temp = []
        ## ip address dest (32 binary )###

        ip_addr = line[5].strip()

        if ip_addr == 'DNS':
            ip = '192.168.63.1'
            ip_data.append(ip)

        elif len(ip_addr) < 10:
            tail = ip_addr.split('_')[1]
            ip = "192.168.100." + tail
            ip_data.append(ip)




        elif ip_addr == "EXT_SERVER":
            ip = '0.0.0.0'
            ip_data.append(ip)


        else:
            ip_data.append(ip_addr)

        ### proto ###

        proto.append(line[2].strip())

        ### dest port (devided by maximum port number  65534)
        try:
            dest_port.append(line[6])
        except ValueError:
            continue

        ### duration, pack, byte, flow  >>>  continuous ###
        temp.append(line[1])
        temp.append(line[7])
        temp.append(line[8])
        temp.append(line[9])
        temp = minmax_scale(temp, feature_range=(0, 1))

        attrs.append(temp)

    # ip_data = np.array(ip_data)
    # proto = np.array(proto)
    # dest_port = np.array(dest_port)

    # attrs = np.array(attrs)

    categorical = []
    words = []

    for i in range(len(ip_data)):
        temp = []
        temp.append(ip_data[i])
        temp.append(proto[i])
        temp.append(dest_port[i])
        categorical.append(temp)

        for ele in temp:
            words.append(ele)

    # words = set(words)
    # f = open('./test.txt', 'w')
    # f.write(', '.join(words))
    # f.close()

    words = set(words)
    word_dict = {}

    i = 0
    for word in words:
        word_dict[word.strip()] = i
        i += 1

    vocab_size = len(words) + 1

    print(len(words))
    print("test : ", word_dict['TCP'])

    categorical2 = []

    for item in categorical:
        try:
            temp = []
            for ele in item:
                temp.append(word_dict[ele])
            categorical2.append(temp)
        except KeyError:
            continue

    print("vocab size : ", vocab_size)
    return categorical2, attrs



    print("normal count: ", normal)
    print("attack count : ", attack)
    print("count : ", count)
    print(len(ip_data), len(attrs), len(label))  # 1048575
    return np.array(ip_data), np.array(attrs), np.array(label)


def prepare_testset_with_label2():
    """CIDDS-001 week1.csv / attacker = 850870 / normal = 1048575 """

    file = open("/home/selab/mj/GANTest/dataset/CIDDS-001(flow-based)/traffic/OpenStack/CIDDS-001-internal-week1_modified.csv", 'r')
    rdr = csv.reader(file)

    ### attributes : [ iip dest, proto, dest port, duration ]
    proto = []
    dest_port = []
    attrs = []
    label = []

    normal = 0
    attack = 0
    count = 0

    for idx, line in enumerate(rdr):

        # labeling
        if line[12] == 'normal':
            label.append(0)
            count += 1
            normal += 1
        elif line[12] == 'attacker':
            label.append(1)
            count += 1
            attack += 1
        else:
            continue


        temp = []
        ## ip address dest (32 binary )###


        ### proto ###

        proto.append(line[2].strip())

        ### dest port (devided by maximum port number  65534)
        try:
            dest_port.append(line[6])
        except ValueError:
            continue

        ### duration, pack, byte, flow  >>>  continuous ###
        temp.append(line[1])
        temp.append(line[7])
        temp.append(line[8])
        temp.append(line[9])
        temp = minmax_scale(temp, feature_range=(0, 1))

        attrs.append(temp)

    # ip_data = np.array(ip_data)
    # proto = np.array(proto)
    # dest_port = np.array(dest_port)

    # attrs = np.array(attrs)

    categorical = []
    words = []

    for i in range(len(proto)):
        temp = []
        temp.append(proto[i])
        temp.append(dest_port[i])
        categorical.append(temp)

        for ele in temp:
            words.append(ele)

    # words = set(words)
    # f = open('./test.txt', 'w')
    # f.write(', '.join(words))
    # f.close()

    words = set(words)
    word_dict = {}
    vocab_size = len(words)


    i = 0
    for word in words:
        j = i
        word_dict[word.strip()] = i/vocab_size
        i += 1


    print(vocab_size)
    print("test : ", word_dict['TCP'])

    categorical2 = []

    for item in categorical:
        try:
            temp = []
            for ele in item:
                temp.append(word_dict[ele])
            categorical2.append(temp)
        except KeyError:
            continue

    print("vocab size : ", vocab_size)



    print("normal count: ", normal)
    print("attack count : ", attack)
    print("count : ", count)

    dataset = np.hstack([attrs, categorical2])

    print("data shape : ", dataset.shape)
    return dataset, label


def get_test_data():
    file = open(
        "/home/selab/mj/GANTest/dataset/CIDDS-001(flow-based)/traffic/OpenStack/CIDDS-001-internal-week1_modified.csv",
        'r')

    rdr = csv.reader(file)

    ### attributes : [ iip dest, proto, dest port, duration ]
    ip_data = []
    proto = []
    dest_port = []

    duration = []
    packet = []
    byte = []
    flow = []
    label = []

    normal = 0
    attack = 0
    count = 0

    for idx, line in enumerate(rdr):

        # labeling
        if line[12] == 'normal':
            label.append(1)
            count += 1
            normal += 1
        elif line[12] == 'attacker':
            label.append(0)
            count += 1
            attack += 1
        else:
            continue


        ### ip address ###

        ip = line[5].strip()
        ip_data.append(ip)

        ### proto ###

        proto.append(line[2])

        ### dest port (devided by maximum port number  65534)
        try:
            dest_port.append(line[6])
        except ValueError:
            continue

        ### duration, pack, byte, flow  >>>  continuous ###
        duration.append(line[1])
        packet.append(line[7])
        byte.append(line[8])
        # flow.append(line[9])

    file.close()

    ### ip, proto, port (categorical) ###

    ip_set = set(ip_data)
    ip_dict = {}
    i = 1
    for ip in ip_set:
        ip_dict[ip] = i
        i += 1

    ip_value = minmax_scale(list(ip_dict.values()), feature_range=(0, 1))
    i = 0
    for ipp in ip_dict.items():
        ip_dict[ipp[0]] = ip_value[i]
        i += 1

    new_ip_data = []
    for ip_ in ip_data:
        new_ip_data.append(ip_dict[ip_])

    del ip_data
    del ip_dict

    proto_data = set(proto)
    proto_dict = {}
    i = 1
    for pro in proto_data:
        proto_dict[pro] = i
        i += 1

    proto_value = minmax_scale(list(proto_dict.values()), feature_range=(0, 1))
    i = 0
    for prot in proto_dict.items():
        proto_dict[prot[0]] = proto_value[i]
        i += 1

    new_proto = []
    for p in proto:
        new_proto.append(proto_dict[p])

    port_data = set(dest_port)
    port_dict = {}
    i = 1
    for port in port_data:
        port_dict[port] = i
        i += 1

    port_value = minmax_scale(list(port_dict.values()), feature_range=(0, 1))
    i = 0
    for portt in port_dict.items():
        port_dict[portt[0]] = port_value[i]
        i += 1

    new_port = []
    for p in dest_port:
        new_port.append(port_dict[p])

    # duation, packet, byte, flow (continuous)

    duration = minmax_scale(duration, feature_range=(0, 1))
    packet = minmax_scale(packet, feature_range=(0, 1))
    byte = minmax_scale(byte, feature_range=(0, 1))
    # flow = minmax_scale(flow, feature_range=(0, 1))

    print(len(new_ip_data), len(new_proto), len(duration))

    dataset = []
    for i in range(len(duration)):
        temp = []
        temp.append(new_ip_data[i])
        temp.append(new_proto[i])
        temp.append(new_port[i])

        temp.append(duration[i])
        temp.append(packet[i])
        temp.append(byte[i])
        # temp.append(flow[i])

        dataset.append(temp)

    print("normal : ", normal)
    print("attack : ", attack)
    print("whole : ", len(label))
    dataset = np.array(dataset)

    return dataset, label







def main():
    load_data_4()

if __name__ == '__main__':
    main()
# data_file.write(temp_text)
# data_file.close()



