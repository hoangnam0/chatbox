import random
import time
import json
import torch

from nltk_utils import bag_of_words, tokenize

import torch.nn as nn
import json

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No activation and no softmax
        return out
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open('tes.json', 'r', encoding='utf8') as f:
    intents = json.load(f)

FILE = 'tes.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
product = None
#khởi tạo biến cục bộ tag
current_tag = []

def get_response(msg):
    
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    
    
    if prob.item() > 0.75:
       
        for intent in intents['intents']:
            
            #Tư vấn samsung
            if   "smarphonesystemA" in current_tag  and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(cungduoc)" in current_tag and "smarphonepin(4.000mAh)" in current_tag  and tag=="smarphonehang(Samsung)":
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: 1. Samsung S7 \n 2. Samsung A7";
            if "smarphonesystemA" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(cungduoc)" in current_tag and "smarphonepin(5.000mAh)" in current_tag  and tag=="smarphonehang(Samsung)":
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: Samsung s8"
            if "smarphonesystemA" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(32GB)" in current_tag and "smarphonecamera(cungduoc)" in current_tag and "smarphonepin(4.000mAh)" in current_tag  and tag=="smarphonehang(Samsung)":
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: Samsung s9.1</br> <img src='../images/logo.png'>"
            if "smarphonesystemA" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(32GB)" in current_tag and "smarphonecamera(cungduoc)" in current_tag and "smarphonepin(5.000mAh)" in current_tag  and tag=="smarphonehang(Samsung)":
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: Samsung s9.2"
            if "smarphonesystemA" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(32GB)" in current_tag and "smarphonecamera(tot)" in current_tag and "smarphonepin(5.000mAh)" in current_tag and tag=="smarphonehang(Samsung)":
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: Samsung note 9.2 (32GB)"
            if "smarphonesystemA" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(32GB)" in current_tag and "smarphonecamera(tot)" in current_tag and "smarphonepin(4.000mAh)" in current_tag and tag=="smarphonehang(Samsung)":
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: Samsung note 9.1 (32GB)"
            if "smarphonesystemA" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(tot)" in current_tag and "smarphonepin(4.000mAh)" in current_tag  and tag=="smarphonehang(Samsung)":
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: Samsung note 8.1 (16GB)"
            if "smarphonesystemA" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(tot)" in current_tag and "smarphonepin(5.000mAh)" in current_tag  and tag=="smarphonehang(Samsung)":
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của samsung như: Samsung note 8.2 (16GB)"
            # Tư vấn iphone
            if "smarphonesystemI" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(cungduoc)" in current_tag and tag == "smarphonepin(4.000mAh)" :
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của ios như: Iphone 5 (16GB)"
            if "smarphonesystemI" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(tot)" in current_tag and tag == "smarphonepin(4.000mAh)" :
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của ios như: Iphone 5 plus (16GB)"
            if "smarphonesystemI" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(cungduoc)" in current_tag and tag == "smarphonepin(5.000mAh)" :
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của ios như: Iphone 6 (16GB)"
            if "smarphonesystemI" in current_tag and "smarphoneprice(1-4)" in current_tag and "smarphoneMemory(16GB)" in current_tag and "smarphonecamera(tot)" in current_tag and tag == "smarphonepin(5.000mAh)" :
                current_tag.append("iphone6plus")
                print (current_tag)
                return "Dạ hiện tại để đáp ứng các nhu cầu của bạn với các thông tin ở trên thì shop muốn tư vấn cho bạn về các sản phẩm của ios như: Iphone 6 plus (16GB)</br> Với sản phẩm Iphone 6 plus bên shop mình có bảng màu:</br> 1. Màu đỏ </br> 2. Màu Vàng"
            if "iphone6plus" in current_tag and tag== "color(red)":
                return "Dạ hiện màu đỏ iphone 6 plus bên shop mình vẫn còn hàng ạ. Bạn có muốn tôi hướng dẫn cách đặt hàng online không?"
            if  "điện thoại" in sentence and "rẻ" in sentence and "rẻ nhất" in sentence or "giá" in sentence:
                current_tag.clear() 
                return "Bạn muốn điện thoại sử dụng hệ điều hành nào:</br> 1. Android</br> 2. iOS"
            if  tag== "tuvanlai":
                current_tag.clear()
                return "Bạn muốn điện thoại sử dụng hệ điều hành nào:</br> 1. Android</br> 2. iOS"

            if tag == intent["tag"]: 
                current_tag.append(tag)
                print (current_tag)
                return random.choice(intent['responses']);  
        else:
            return "Toi không hiểu bạn nói gì"    
            
                
                
                
    else:
        return "Tôi không hiểu bạn nói gì? Vui lòng làm theo hướng dẫn!!"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        chat_history = [] # tạo một list rỗng để lưu lịch sử chat
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        chat_history.append(resp) # thêm câu trả lời mới vào lịch sử chat
        print(resp)
        print("Chat history:", chat_history) # in ra lịch sử chat