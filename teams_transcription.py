import pyaudio
import wave
import threading
import time
import whisper
import numpy as np
from datetime import datetime
import os
import queue
import logging

class TeamsTranscriptionApp:
    def __init__(self, output_dir="transcriptions"):
        """
        Teams会議音声のリアルタイム文字起こしアプリ
        
        Args:
            output_dir (str): 出力ファイルを保存するディレクトリ
        """
        self.output_dir = output_dir
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # 音声設定
        self.chunk_size = 1024
        self.sample_rate = 16000
        self.channels = 1
        self.audio_format = pyaudio.paInt16
        
        # Whisperモデルの初期化（小さいモデルで高速化）
        print("Whisperモデルを読み込んでいます...")
        self.model = whisper.load_model("base")
        print("モデルの読み込み完了")
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 出力ファイルの準備
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(output_dir, f"meeting_transcript_{timestamp}.txt")
        
    def list_audio_devices(self):
        """利用可能な音声デバイスを一覧表示"""
        p = pyaudio.PyAudio()
        print("\n=== 利用可能な音声デバイス ===")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']} (入力: {info['maxInputChannels']}, 出力: {info['maxOutputChannels']})")
        p.terminate()
        
    def get_default_input_device(self):
        """デフォルトの入力デバイスを取得"""
        p = pyaudio.PyAudio()
        default_device = p.get_default_input_device_info()
        p.terminate()
        return default_device['index']
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音声データのコールバック関数"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def transcribe_audio_chunk(self, audio_data):
        """音声チャンクを文字起こし"""
        try:
            # バイトデータをnumpy配列に変換
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 音声の長さが短すぎる場合はスキップ
            if len(audio_np) < self.sample_rate * 0.5:  # 0.5秒未満
                return None
                
            # Whisperで文字起こし
            result = self.model.transcribe(audio_np, language='ja')
            text = result['text'].strip()
            
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"文字起こしエラー: {e}")
            return None
    
    def transcription_worker(self):
        """文字起こし処理のワーカースレッド"""
        audio_buffer = b''
        buffer_duration = 3.0  # 3秒のバッファ
        buffer_size = int(self.sample_rate * buffer_duration * 2)  # 16bit = 2bytes
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # キューから音声データを取得
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=1.0)
                    audio_buffer += audio_data
                
                # バッファが十分たまったら文字起こし実行
                if len(audio_buffer) >= buffer_size:
                    text = self.transcribe_audio_chunk(audio_buffer)
                    
                    if text:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        output_line = f"[{timestamp}] {text}\n"
                        
                        # コンソール出力
                        print(f"🎤 {output_line.strip()}")
                        
                        # ファイル出力
                        with open(self.output_file, 'a', encoding='utf-8') as f:
                            f.write(output_line)
                            f.flush()  # リアルタイムでファイルに書き込み
                    
                    # バッファをクリア（オーバーラップ用に少し残す）
                    overlap = buffer_size // 4
                    audio_buffer = audio_buffer[-overlap:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"文字起こしワーカーエラー: {e}")
    
    def start_recording(self, device_index=None):
        """録音開始"""
        try:
            p = pyaudio.PyAudio()
            
            if device_index is None:
                device_index = self.get_default_input_device()
            
            print(f"音声デバイス {device_index} を使用します")
            print(f"出力ファイル: {self.output_file}")
            print("録音開始... (Ctrl+Cで停止)")
            
            # 出力ファイルにヘッダーを書き込み
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Teams会議 文字起こしログ ===\n")
                f.write(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
            
            # 録音開始
            self.is_recording = True
            
            # 文字起こしワーカースレッド開始
            transcription_thread = threading.Thread(target=self.transcription_worker)
            transcription_thread.start()
            
            # 音声ストリーム開始
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            stream.start_stream()
            
            # メインループ
            try:
                while stream.is_active():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n録音を停止しています...")
            
            # 録音停止
            self.is_recording = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # ワーカースレッドの終了を待機
            transcription_thread.join(timeout=10)
            
            # 終了メッセージをファイルに書き込み
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"録音完了。ファイルが保存されました: {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"録音エラー: {e}")
            self.is_recording = False

def main():
    """メイン関数"""
    app = TeamsTranscriptionApp()
    
    print("=== Teams会議音声文字起こしアプリ ===")
    print("\n使用方法:")
    print("1. Teamsで会議に参加")
    print("2. このアプリを起動")
    print("3. 音声デバイスを選択（通常はデフォルトでOK）")
    print("4. 録音開始")
    print("\n注意: システムの音声出力（スピーカー音）を録音するため、")
    print("      適切な音声設定が必要です。")
    
    # 音声デバイス一覧表示
    app.list_audio_devices()
    
    # デバイス選択
    try:
        device_input = input("\n使用する音声デバイス番号を入力 (Enterでデフォルト): ").strip()
        device_index = int(device_input) if device_input else None
    except ValueError:
        device_index = None
    
    # 録音開始
    app.start_recording(device_index)

if __name__ == "__main__":
    main()