from fbmessenger import BaseMessenger
from fbmessenger import MessengerClient
from fbmessenger.quick_replies import QuickReplies, QuickReply

import os
from flask import Flask, request
# from underthesea import ner
# from intent_module_3 import *
# from ner_module import *
from response_system import *
from underthesea import word_tokenize

app = Flask(__name__)
ACCESS_TOKEN = 'EAAPaSI013YcBAGf7emwvC30YlFXqbgikJvZARZBtr3oVGZA1qX8dhfFsafSuyhnCSxJYBR3UyVZAWT4ia0UH6ZCyQM4kmA1KcAx5TkQZBhZB7dOr8MfvS6MD2RkItToUyZA2zhV0dk8qOCXfs6yWwGUCMDelZCKYZBSAwZBSQHFEcYjSv1IeLcZBPOcYL5doLnTnOCMZD'
VERIFY_TOKEN = 'VERIFY_TOKEN'
# app.debug = False

class Messenger(BaseMessenger):
    def __init__(self, page_access_token, app_secret=None):
        self.page_access_token = page_access_token
        self.app_secret = app_secret
        self.client = MessengerClient(self.page_access_token, app_secret=self.app_secret)

    def message(self, message):
        # self.send({'text': 'Received: {0}'.format(message['message']['text'])}, 'RESPONSE')
        pass

    def delivery(self, message):
        pass

    def read(self, message):
        pass

    def account_linking(self, message):
        pass

    def postback(self, message):
        pass

    def optin(self, message):
        pass


import os
from flask import Flask, request

app = Flask(__name__)
app.debug = True

messenger = Messenger(ACCESS_TOKEN)


loc_apply_flag = False

@app.route('/', methods=['GET', 'POST'])
def receive_message():
    global intention
    intention = 0
    global loc_apply_flag

    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook.""" 
        if (request.args.get('hub.verify_token') == VERIFY_TOKEN):
            return request.args.get('hub.challenge')
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
    # get whatever message a user sent the bot
        output = request.get_json()
        print('__check output variable:', output)
        messenger.handle(request.get_json(force=True))
        inten_flow(intention)
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                #Facebook Messenger ID for user so we know where to send response back to
                    if message['message'].get('text'):
                        print('__check message variable:', message)
                        if  'quick_reply' in message['message']:
                            text = {'text': message['message']['quick_reply']['payload']}

                            inten_flow(execute_flow(message['message']['quick_reply']['payload'], intention))
                            text['quick_replies'] = quick_replies.to_dict()
                            messenger.send(text, 'RESPONSE')
                        else:
                            # print('message type: ', type(message['message'].get('text')))
                            # response_message = repOfficial(message['message'].get('text'))
                            
                            # text = {'text': response_message}
                            # # # print("TEXT: ", text)
                            # # text = {'text': 'A message Hi'}
                            # # # text['quick_replies'] = quick_replies.to_dict()
                            # # messenger.send('Bot say: {0}'.format(text['text']), 'RESPONSE')                            
                            # # text = {'text': 'A Message'}
                            # # text['quick_replies'] = quick_replies.to_dict()
                            # messenger.send(text, 'RESPONSE')   

                            message_text = message['message'].get('text')
                            list_of_out_of_work = ['cmnd', 'chứng minh', 'hộ khẩu', 'KT1', 'KT2', 'KT3']
                            list_of_say_hello = ['hello', 'hi', 'chào', 'aloha', 'morning']
                            message_check = word_tokenize(message_text)
                            for i in range(len(message_check)):
                                if message_check[i] in list_of_out_of_work:
                                    str_bot_rep = "Xin lỗi bạn, tôi chỉ có thể hỗ trợ bạn về vấn đề hộ chiếu, những thủ tục liên quan khác. \
                                    Bạn xin chờ tính năng phát triển tiếp theo."
                                    text = {'text': str_bot_rep}
                                    messenger.send(text, 'RESPONSE') 
                                    return ''
                                elif message_check[i] in list_of_say_hello:
                                    str_bot_rep = "Chào bạn, tôi là Chatbot hỗ trợ bạn với những thủ tục cơ bản khi làm hộ chiếu lần đầu.\
                                    Nếu bạn có thắc mắc gì về những việc cần làm khi làm hộ chiếu lần đầu thì cứ hỏi tôi."
                                    text = {'text': str_bot_rep}
                                    messenger.send(text, 'RESPONSE') 
                                    return ''

                            bot_rep = []
                            userIntent = ic_predict(message_text)[0]
                            print("user's intent: ", userIntent)
                            print('loc_apply_flag: ', loc_apply_flag)

                            if loc_apply_flag == True:
                                locQuantity, locApply = getNerName('Làm hộ chiếu ở ' + message_text)
                                print('locApply: ', locApply)
                                for i in range(len(locQuantity)):
                                    bot_rep.append(getLocApply(locApply[i]))
                                loc_apply_flag = False
                            else:
                                if userIntent == "where_loc_apply":
                                    if isAnyLOC(message_text) == False:
                                        text = {'text': 'Bạn đang ở tỉnh thành: '}
                                        messenger.send(text, 'RESPONSE') 
                                        loc_apply_flag = True
                                        return ''
                                    else:
                                        locQuantity, locApply = getNerName(message_text)
                                        print(locApply)
                                        if locApply == []:
                                            bot_rep = getLocApply(locApply)
                                        else:
                                            for i in range(len(locQuantity)):
                                                bot_rep.append(getLocApply(locApply[i]))
                                else:
                                    bot_rep = normalRep(userIntent)

                            str_bot_rep = ''
                            for i in range(len(bot_rep)):
                                str_bot_rep = str_bot_rep + bot_rep[i]                            

                            text = {'text': str_bot_rep} #We must add 'text' variable like this or it can not send to Messenger.
                            print('text: ', text)
                            messenger.send(text, 'RESPONSE') 
                            print('message sent')
  
    return ''
def inten_flow(option):
    global quick_replies
    if option == 0 :
        quick_reply_1 = QuickReply(title='Go to option 1', payload='Came to option 1')
        quick_reply_2 = QuickReply(title='Go to option 2', payload='Came to option 2')
        quick_reply_3 = QuickReply(title='Go to option 3', payload='Came to option 3')
        quick_replies = QuickReplies(quick_replies=[
            quick_reply_1,
            quick_reply_2,
            quick_reply_3
        ])
    if option == 1:
        quick_reply_1 = QuickReply(title='Go back', payload='Back to option 0')
        quick_replies = QuickReplies(quick_replies=[
            quick_reply_1
        ])
    if option == 2 :
        quick_reply_1 = QuickReply(title='Go to option 4', payload='Came to option 4')
        quick_reply_2 = QuickReply(title='Go to option 6', payload='Came to option 6')
        quick_reply_3 = QuickReply(title='Go back', payload='Back to option 0')
        quick_replies = QuickReplies(quick_replies=[
            quick_reply_1,
            quick_reply_2,
            quick_reply_3
        ])
    if option == 3 :
        quick_reply_1 = QuickReply(title='Go to option 5', payload='Came to option 5')
        quick_reply_2 = QuickReply(title='Go back', payload='Back to option 0')
        quick_replies = QuickReplies(quick_replies=[
            quick_reply_1,
            quick_reply_2
        ])
    if option == 4:
        quick_reply_1 = QuickReply(title='Go back', payload='Back to option 2')
        quick_replies = QuickReplies(quick_replies=[
            quick_reply_1
        ])
    if option == 5:
        quick_reply_1 = QuickReply(title='Go back', payload='Back to option 3')
        quick_replies = QuickReplies(quick_replies=[
            quick_reply_1
        ])
    if option == 6:
        quick_reply_1 = QuickReply(title='Go back', payload='Back to option 2')
        quick_replies = QuickReplies(quick_replies=[
            quick_reply_1
        ])

def execute_flow(intent, option):
    list_of_intent = ['Came to option 1',
                    'Came to option 2',
                    'Came to option 3',
                    'Came to option 4',
                    'Came to option 5',
                    'Came to option 6',
                    'Back to option 0',
                    'Back to option 2',
                    'Back to option 3',]
    for inten in list_of_intent:
        if intent == inten:
            option = int(inten[-1:])
    return option
        
if __name__ == "__main__":
    app.run()
# quick_reply_1 = QuickReply(title='Do something', payload='Send me this payload')
# quick_reply_2 = QuickReply(title='Do something else', payload='Send me this other payload')
# quick_replies = QuickReplies(quick_replies=[
#     quick_reply_1,
#     quick_reply_2
# ])
# # text = { 'text': 'A Message'}
# text = quick_replies.to_dict()
# print(text)
# print(text[1]['payload'])
# message_n = {'sender': {'id': '2225825704162394'},
#  'recipient': {'id': '984518385084498'},
#   'timestamp': 1558405767463, 'message':
#    {'mid': 'f5BwsUsGS5LkR59dYMts7TPqaclJ0j6PF3t3JjCeKPSJZoKjVQImlQnlBWcykhf6K_A9lfw3GeMfBFxR8WQoOw', 'seq': 0, 'text': 'yo'}}
# check = message_n['message']
# if 'quick_replies' in check:
#     print(True)
# print(message_n['message'])
# message = {'sender': {'id': '2225825704162394'}, 
# 'recipient': {'id': '984518385084498'},
#  'timestamp': 1558405210630,
#   'message': {'quick_reply': {'payload': 'Send me this payload'}, 'mid': 'ojiIAq_YcO8aW77ySPo2ajPqaclJ0j6PF3t3JjCeKPTYN-nWQ3qQk1ZUhNqzR9gQFTTG_upjXzucvTUbqzDnTQ', 'seq': 0, 'text': 'Do something'}}
# if 'quick_reply' in message['message']:
#     print(True)
# print(message['message']['quick_reply']['payload'])