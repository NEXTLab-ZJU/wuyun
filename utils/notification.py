import json
import requests
import datetime
import sys


# feishu wuyun chat robot (your code)
api_url = "https://open.feishu.cn/open-apis/bot/v2/hook/XXXX"
headers = {'Content-Type': 'application/json;charset=utf-8'}
curr_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(curr_time,'%H:%M:%S')



def send_msg(text):
    json_text={
        "msg_type":"text",
        "content":{
            "msg":"Message:",
            "text":text
        }
    }
    requests.post(api_url,json.dumps(json_text),headers=headers).content

if __name__ == '__main__':
    message = time_str
    send_msg(message)



	
    
    