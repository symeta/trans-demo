from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import os
import threading
import queue
import pyaudio
import json
import boto3
import numpy as np
import asyncio
from flask_socketio import SocketIO
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
recording = False
audio_stream = None
transcription_thread = None

"""
实时语音转文字处理类，处理来自Amazon Transcribe的响应
"""
class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_result_stream):
        super().__init__(transcript_result_stream)
        self.last_result = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # 提取最新的转录结果
        results = transcript_event.transcript.results
        if len(results) > 0 and results[0].is_partial == False:
            transcript = results[0].alternatives[0].transcript
            if transcript != self.last_result:
                self.last_result = transcript
                print(f"\r转录结果: {transcript}", flush=True)
                # 通过Socket.IO将转录结果发送到前端
                socketio.emit('transcription_update', {'text': transcript})

"""
音频流类，负责从麦克风捕获音频并发送到Amazon Transcribe
"""
class AudioStream:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        print("麦克风已启动，开始说话...")
    
    def read_audio(self):
        return self.stream.read(self.chunk, exception_on_overflow=False)
    
    def close(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p') and self.p:
            self.p.terminate()
        print("音频流已关闭")

"""
转录服务的主函数
"""
async def transcribe_audio():
    global recording, audio_stream
    
    # 初始化Amazon Transcribe streaming client
    client = TranscribeStreamingClient(region="us-east-1")  # 修改为您的区域
    
    # 配置流式转录请求
    stream = await client.start_stream_transcription(
        language_code="zh-CN",  # 可根据需要更改语言
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )
    
    # 设置事件处理程序
    handler = MyEventHandler(stream.output_stream)
    
    # 启动处理响应的协程
    process_task = asyncio.create_task(handler.handle_events())
    
    # 初始化音频流
    audio_stream = AudioStream()
    
    try:
        print("转录服务已启动...")
        while recording:
            # 读取音频数据并发送到Transcribe
            audio_chunk = audio_stream.read_audio()
            await stream.input_stream.send_audio_event(audio_chunk=audio_chunk)
            
            # 短暂暂停，避免CPU负担过重
            await asyncio.sleep(0.001)
    
    except Exception as e:
        print(f"\n转录过程中发生错误: {str(e)}")
    finally:
        # 清理资源
        await stream.input_stream.end_stream()
        await process_task
        if audio_stream:
            audio_stream.close()
            audio_stream = None
        print("转录服务已结束")

def run_transcription():
    asyncio.run(transcribe_audio())

class VideoRecorder:
    def __init__(self):
        self.cap = None
        self.out = None
        self.recording = False
        self.frame = None
        self.output_file = None
        self.record_thread = None

    def start_recording(self):
        if self.recording:
            return {"status": "already_recording"}

        # 获取当前时间作为文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = f"recorded_video_{timestamp}.mp4"
        
        # 尝试打开摄像头
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            return {"status": "error", "message": "无法打开摄像头"}
        
        # 获取视频帧的宽度和高度
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置视频编解码器和FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        fps = 30.0
        
        # 创建VideoWriter对象
        self.out = cv2.VideoWriter(self.output_file, fourcc, fps, (width, height))
        
        self.recording = True
        
        # 在新线程中开始录制
        self.record_thread = threading.Thread(target=self.record)
        self.record_thread.daemon = True
        self.record_thread.start()
        
        return {"status": "success", "file": self.output_file}

    def record(self):
        while self.recording:
            ret, frame = self.cap.read()
            if not ret:
                self.recording = False
                break
                
            self.frame = frame
            self.out.write(frame)
            time.sleep(0.01)  # 降低CPU使用率

    def get_frame(self):
        if self.cap is None or self.frame is None:
            # 返回一个黑色的帧
            black_frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
            return black_frame
            
        # 将帧编码成jpg格式
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()

    def stop_recording(self):
        if not self.recording:
            return {"status": "not_recording"}
            
        self.recording = False
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
            
        # 释放资源
        if self.out:
            self.out.release()
            self.out = None
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        file_path = os.path.abspath(self.output_file) if self.output_file else None
        
        return {"status": "success", "file": file_path}

# 实例化录像机
recorder = VideoRecorder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not recorder.recording and recorder.cap is None:
                # 如果没有录制，返回黑色帧
                black_frame = cv2.imencode('.jpg', 
                                          cv2.putText(
                                              np.zeros((480, 640, 3), dtype=np.uint8), 
                                              "摄像头未启动", 
                                              (160, 240), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 
                                              1, 
                                              (255, 255, 255), 
                                              2)
                                          )[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + black_frame + b'\r\n')
                time.sleep(0.1)
                continue
                
            frame = recorder.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
            
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, transcription_thread
    
    # 设置全局录制标志
    recording = True
    
    # 启动视频录制
    result = recorder.start_recording()
    if result["status"] != "success":
        recording = False
        return jsonify(result)
    
    # 启动转录服务的线程
    if transcription_thread is None:
        transcription_thread = threading.Thread(target=run_transcription)
        transcription_thread.daemon = True
        transcription_thread.start()
    
    return jsonify(result)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, transcription_thread, audio_stream
    
    # 停止录制
    recording = False
    
    # 停止视频录制
    result = recorder.stop_recording()
    
    # 等待转录线程结束
    if transcription_thread:
        transcription_thread.join(timeout=2)
        transcription_thread = None
    
    # 确保音频流关闭
    if audio_stream:
        audio_stream.close()
        audio_stream = None
    
    return jsonify(result)

if __name__ == '__main__':
    socketio.run(app, debug=True)