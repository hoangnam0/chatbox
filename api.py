import random
import time
import json
import torch

from nltk_utils import bag_of_words, tokenize
import fastapi
import uvicorn
from model_api.message import Message
import torch
import torch.nn as nn

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

#device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open('tes.json', 'r', encoding="utf-8") as f:
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

app = fastapi.FastAPI()
@app.post("/shoe/chatbot")

async def get_response(messenge: Message):
    
    if messenge is None or messenge.msg == "quit":
        return {"response": "Bye", "status": "success"}
    try:
        sentence = messenge.msg

        sentence = tokenize(sentence)
        x = bag_of_words(sentence, all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(device)

        output = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        result = ""
        # if prob.item() > 0.75:
        #     for intent in intents['intents']:
        #         if tag == intent["tag"]:
        #             result = random.choice(intent['responses'])
        #             print(result)
        
        # else:
        #     result = "I do not understand..."
        #     print(result)
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if tag == "smartphone":
                        result = random.choice(intent['responses'])
                        print(result)
                        user_input = input('You: ')
                        user_input = tokenize(user_input)
                        x1 = bag_of_words(user_input, all_words)
                        x1 = x1.reshape(1, x1.shape[0])
                        x1 = torch.from_numpy(x1).to(device)

                        output1 = model(x1)
                        _, predicted = torch.max(output1, dim=1)
                        tag1 = tags[predicted.item()]
                        
                        for sub_intent in intents['intents']:
                            if tag1 == sub_intent["tag"]:
                                if tag1 == "samsung":
                                    result = random.choice(sub_intent['responses'])
                                    print(result)
                                    user_input1 = input('You: ')
                                    user_input1 = tokenize(user_input1)
                                    x2 = bag_of_words(user_input1, all_words)
                                    x2 = x2.reshape(1, x2.shape[0])
                                    x2 = torch.from_numpy(x2).to(device)

                                    output2 = model(x2)
                                    _, predicted = torch.max(output2, dim=1)
                                    tag2 = tags[predicted.item()]
                                    for sub_intent1 in intents['intents']:
                                        if tag2== sub_intent1["tag"] and tag1=="samsung":
                                            if tag2 == "samsung_price(1-4)":
                                                result = random.choice(sub_intent1['responses'])
                                                print(result)
                                                user_input = input('You: ')
                                            if tag2 == "samsung_price(4-8)":
                                                # print(f"{bot_name}: {random.choice(sub_intent1['responses'])}")
                                                result = random.choice(sub_intent1['responses'])
                                                print(result)
                                                user_input = input('You: ')
                                                break
                                if tag1 == "Iphone":
                                    
                                    result = random.choice(sub_intent['responses'])
                                    print(result)
                                    user_input1 = tokenize(user_input1)
                                    x2 = bag_of_words(user_input1, all_words)
                                    x2 = x2.reshape(1, x2.shape[0])
                                    x2 = torch.from_numpy(x2).to(device)

                                    output2 = model(x2)
                                    _, predicted = torch.max(output2, dim=1)
                                    tag2 = tags[predicted.item()]
                                    for sub_intent1 in intents['intents']:
                                        if tag2== sub_intent1["tag"]:
                                            if tag2 == "Iphone_price(1-4)":
                                                # print(f"{bot_name}: {random.choice(sub_intent2['responses'])}")
                                                result = random.choice(sub_intent1['responses'])
                                                print(result)
                                                user_input2 = input('You: ')
                                                
                                    #         if tag2 == "Iphone_price(4-8)":
                                    #             print(f"{bot_name}: {random.choice(sub_intent2['responses'])}")
                                    #             user_input = input('You: ')
                                    #             break
                                                
                                                user_input2 = tokenize(user_input2)
                                                x3 = bag_of_words(user_input2, all_words)
                                                x3 = x3.reshape(1, x3.shape[0])
                                                x3 = torch.from_numpy(x3).to(device)
                                                output3 = model(x3)
                                                _, predicted = torch.max(output3, dim=1)
                                                tag3 = tags[predicted.item()]
                                                for sub_intent1 in intents['intents']:
                                                    if tag3== sub_intent1["tag"]:
                                                        if tag3 == "Iphone_color(white)":
                                                            result = random.choice(sub_intent1['responses'])
                                                            print(result)
                                                            # print(f"{bot_name}: {random.choice(sub_intent['responses'])}")
                                                            user_input3 = input('You: ')
                                        
    
                                                    
                                else:
                                    # print(f"{bot_name}: Vui lòng nhập đúng tên hãng")
                                    result = "Vui lòng nhập đúng tên hãng"
                                    print(result)
                    
                    else:
                        # print(f"{bot_name}: I do not understand...")
                        result = "I do not understand..."
                        print(result)
                
        else:
            # print(f"{bot_name}: I do not understand...")
            result = "I do not understand..."
            print(result)
            
    except:
        return {"response": "Oops! Something went wrong!", "status": "failed"}
    
    return {"response": result, "status": "success"}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=3306)