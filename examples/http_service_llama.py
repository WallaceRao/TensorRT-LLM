from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import multiprocessing
import threading
import os
import base64

from datetime import datetime
import logging
import numpy as np
import uuid
import ssl
import json

from run_test import run_infer

ssl._create_default_https_context = ssl._create_unverified_context

tts_logger = logging.getLogger("llm")
tts_logger.setLevel(logging.INFO)
log_path = "/tensorrt-llm/llm.log"
handler = logging.FileHandler(log_path, mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
tts_logger.addHandler(handler)


                    
class Handler(SimpleHTTPRequestHandler):
    def send_post_response(self, response_str):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(response_str.encode("UTF-8"))))
        self.end_headers()
        self.wfile.write(response_str.encode("UTF-8"))
    
    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        req_id = str(uuid.uuid4().fields[-1])[:5]
        tts_logger.info(f"received a request, req_id:{req_id}")
        json_obj = json.loads(data_string)
        err_msg = "OK"
        input_tokens = []
        if "input_tokens" in json_obj.keys() and len(json_obj["input_tokens"]) > 0:
            data_str = json_obj["input_tokens"]
            decoded_binary = base64.b64decode(data_str)
            input_tokens = np.frombuffer(decoded_binary, dtype=np.int32).tolist()
        else:
            err_msg = "no input_tokens received"
            response_str = json.dumps({"err_msg": err_msg})
            self.send_post_response(response_str)
            return
        
        print("input_tokens:", input_tokens)
        output_tokens = run_infer(input_tokens)
        print("output_tokens:", output_tokens)
        output_bytes = np.array(output_tokens, dtype=np.int32).tobytes()
        base64_str = base64.b64encode(output_bytes)
        response_str = json.dumps({"output_tokens": base64_str.decode(), "err_msg":err_msg})
        self.send_post_response(response_str)
        return

def run():
    port = 78
    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print("server started on port:", port)
    server.serve_forever()

if __name__ == '__main__':
    os.environ["WORK_DIR"] = "./"
    run()