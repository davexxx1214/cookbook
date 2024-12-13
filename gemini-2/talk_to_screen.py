# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image
import json
import mss

from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

with open('config.json', 'r') as f:
    config = json.load(f)

MODEL = config['model']
GOOGLE_API_KEY = config['api_key']

client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

CONFIG = {
    "generation_config": {"response_modalities": ["AUDIO"]}
}

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()
        self.video_out_queue = asyncio.Queue()
        self.frame_queue = asyncio.Queue()  # 用于主线程和异步任务之间传递帧数据
        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await self.session.send(text or ".", end_of_turn=True)

    def _get_frame(self, sct):
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)

        if not sct_img:
            return None
        
        img = PIL.Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def capture_frames(self): # 主线程执行
        with mss.mss() as sct:
           while True:
                frame = self._get_frame(sct)
                if frame is None:
                    break
                self.frame_queue.put_nowait(frame)  # 将帧数据放入队列
                await asyncio.sleep(1.0)

    async def get_frames(self): # 异步任务，从队列读取数据
        while True:
            frame = await self.frame_queue.get()
            self.video_out_queue.put_nowait(frame)


    async def send_frames(self):
        while True:
            frame = await self.video_out_queue.get()
            await self.session.send(frame)

    async def listen_audio(self):
        pya = pyaudio.PyAudio()

        mic_info = pya.get_default_input_device_info()
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        while True:
            data = await asyncio.to_thread(stream.read, CHUNK_SIZE)
            self.audio_out_queue.put_nowait(data)

    async def send_audio(self):
        while True:
            chunk = await self.audio_out_queue.get()
            await self.session.send({"data": chunk, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            async for response in self.session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        parts = model_turn.parts

                        for part in parts:
                            if part.text is not None:
                                print(part.text, end="")
                            elif part.inline_data is not None:
                                self.audio_in_queue.put_nowait(part.inline_data.data)

                    server_content.model_turn = None
                    turn_complete = server_content.turn_complete
                    if turn_complete:
                        # If you interrupt the model, it sends a turn_complete.
                        # For interruptions to work, we need to stop playback.
                        # So empty out the audio queue because it may have loaded
                        # much more audio than has played yet.
                        print("Turn complete")
                        while not self.audio_in_queue.empty():
                            self.audio_in_queue.get_nowait()

    async def play_audio(self):
        pya = pyaudio.PyAudio()
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        """Takes audio chunks off the input queue, and writes them to files.

        Splits and displays files if the queue pauses for more than `max_pause`.
        """
        
        async with (
            client.aio.live.connect(model=MODEL, config=CONFIG) as session,
            asyncio.TaskGroup() as tg,
        ):
            self.session = session

            send_text_task = tg.create_task(self.send_text())
            
            def cleanup(task):
                 for t in tg._tasks:
                     t.cancel()
            send_text_task.add_done_callback(cleanup)
            
            tg.create_task(self.listen_audio())
            tg.create_task(self.send_audio())
            tg.create_task(self.get_frames())
            tg.create_task(self.send_frames())
            tg.create_task(self.receive_audio())
            tg.create_task(self.play_audio())
            
            # 在这里启动主线程的帧捕获任务
            asyncio.create_task(self.capture_frames())

            def check_error(task):
                if task.cancelled():
                    return
                if task.exception() is None:
                    return

                e = task.exception()
                traceback.print_exception(None, e, e.__traceback__)
                sys.exit(1)

            for task in tg._tasks:
                task.add_done_callback(check_error)

if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())