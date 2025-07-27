import os
import sys
import whisper
import moviepy.editor as mp
from datetime import datetime, timedelta
import argparse
import logging
from pathlib import Path
import tempfile
import numpy as np

class MP4TranscriptionApp:
    def __init__(self, model_size="base", output_dir="transcriptions"):
        """
        MP4動画ファイルの音声文字起こしアプリ
        
        Args:
            model_size (str): Whisperモデルサイズ (tiny, base, small, medium, large)
            output_dir (str): 出力ファイルを保存するディレクトリ
        """
        self.output_dir = output_dir
        self.model_size = model_size
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Whisperモデルの初期化
        print(f"Whisperモデル ({model_size}) を読み込んでいます...")
        self.model = whisper.load_model(model_size)
        print("モデルの読み込み完了")
    
    def extract_audio_from_mp4(self, mp4_path, audio_path=None):
        """
        MP4ファイルから音声を抽出
        
        Args:
            mp4_path (str): MP4ファイルのパス
            audio_path (str): 出力する音声ファイルのパス
            
        Returns:
            str: 抽出された音声ファイルのパス
        """
        try:
            # 音声ファイルのパスを決定
            if audio_path is None:
                base_name = Path(mp4_path).stem
                audio_path = os.path.join(tempfile.gettempdir(), f"{base_name}_audio.wav")
            
            print(f"音声を抽出中: {mp4_path} -> {audio_path}")
            
            # MoviePyで音声を抽出
            video = mp.VideoFileClip(mp4_path)
            audio = video.audio
            
            # WAVファイルとして保存
            audio.write_audiofile(
                audio_path,
                verbose=False,
                logger=None,  # MoviePyのログを非表示
                temp_audiofile_path=tempfile.gettempdir()
            )
            
            # リソースを解放
            audio.close()
            video.close()
            
            print(f"音声抽出完了: {audio_path}")
            return audio_path
            
        except Exception as e:
            self.logger.error(f"音声抽出エラー: {e}")
            raise
    
    def format_timestamp(self, seconds):
        """
        秒数を時:分:秒形式に変換
        
        Args:
            seconds (float): 秒数
            
        Returns:
            str: HH:MM:SS形式の時刻
        """
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def transcribe_with_timestamps(self, audio_path, language="ja"):
        """
        音声ファイルをタイムスタンプ付きで文字起こし
        
        Args:
            audio_path (str): 音声ファイルのパス
            language (str): 言語コード (ja, en, etc.)
            
        Returns:
            dict: Whisperの結果
        """
        try:
            print("音声を文字起こし中...")
            
            # Whisperで文字起こし実行
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                word_timestamps=True,  # 単語レベルのタイムスタンプ
                verbose=True
            )
            
            print("文字起こし完了")
            return result
            
        except Exception as e:
            self.logger.error(f"文字起こしエラー: {e}")
            raise
    
    def detect_speakers(self, segments, min_pause=2.0):
        """
        簡易的な話者検出（無音区間で話者変更を推定）
        
        Args:
            segments (list): Whisperのセグメント
            min_pause (float): 話者変更と判定する最小無音時間（秒）
            
        Returns:
            list: 話者情報付きセグメント
        """
        if not segments:
            return []
        
        speaker_segments = []
        current_speaker = 1
        
        for i, segment in enumerate(segments):
            # 前のセグメントとの間隔をチェック
            if i > 0:
                prev_end = segments[i-1]['end']
                current_start = segment['start']
                pause_duration = current_start - prev_end
                
                # 長い無音区間があれば話者変更と推定
                if pause_duration >= min_pause:
                    current_speaker += 1
            
            segment_with_speaker = segment.copy()
            segment_with_speaker['speaker'] = current_speaker
            speaker_segments.append(segment_with_speaker)
        
        return speaker_segments
    
    def save_transcript(self, result, mp4_path, include_speakers=True, include_timestamps=True):
        """
        文字起こし結果をファイルに保存
        
        Args:
            result (dict): Whisperの結果
            mp4_path (str): 元のMP4ファイルパス
            include_speakers (bool): 話者情報を含めるか
            include_timestamps (bool): タイムスタンプを含めるか
            
        Returns:
            str: 出力ファイルのパス
        """
        # 出力ファイル名を生成
        base_name = Path(mp4_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{base_name}_transcript_{timestamp}.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # ヘッダー情報
                f.write("=" * 60 + "\n")
                f.write("MP4動画ファイル音声文字起こし結果\n")
                f.write("=" * 60 + "\n")
                f.write(f"元ファイル: {mp4_path}\n")
                f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Whisperモデル: {self.model_size}\n")
                f.write(f"検出言語: {result.get('language', 'unknown')}\n")
                f.write("=" * 60 + "\n\n")
                
                # セグメント情報を話者検出付きで処理
                segments = result['segments']
                if include_speakers:
                    segments = self.detect_speakers(segments)
                
                # 全文を最初に出力
                f.write("【全文】\n")
                f.write("-" * 40 + "\n")
                full_text = result['text'].strip()
                f.write(full_text + "\n\n")
                
                # セグメント別詳細
                f.write("【詳細（タイムスタンプ付き）】\n")
                f.write("-" * 40 + "\n")
                
                for segment in segments:
                    start_time = self.format_timestamp(segment['start'])
                    end_time = self.format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    
                    if include_timestamps and include_speakers:
                        speaker_info = f"話者{segment.get('speaker', '?')}"
                        f.write(f"[{start_time} - {end_time}] {speaker_info}: {text}\n")
                        print(f"[{start_time} - {end_time}] {speaker_info}: {text}")
                    elif include_timestamps:
                        f.write(f"[{start_time} - {end_time}] {text}\n")
                        print(f"[{start_time} - {end_time}] {text}")
                    else:
                        f.write(f"{text}\n")
                        print(text)
                
                # 統計情報
                f.write("\n" + "=" * 60 + "\n")
                f.write("【統計情報】\n")
                f.write(f"総発話時間: {self.format_timestamp(segments[-1]['end'] if segments else 0)}\n")
                f.write(f"セグメント数: {len(segments)}\n")
                if include_speakers:
                    max_speaker = max([s.get('speaker', 1) for s in segments]) if segments else 1
                    f.write(f"推定話者数: {max_speaker}\n")
                f.write(f"総文字数: {len(full_text)}\n")
                
            print(f"\n✅ 文字起こし結果を保存しました: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"ファイル保存エラー: {e}")
            raise
    
    def process_mp4(self, mp4_path, language="ja", include_speakers=True, include_timestamps=True, keep_audio=False):
        """
        MP4ファイルを処理してテキストファイルを生成
        
        Args:
            mp4_path (str): MP4ファイルのパス
            language (str): 言語コード
            include_speakers (bool): 話者情報を含めるか
            include_timestamps (bool): タイムスタンプを含めるか
            keep_audio (bool): 抽出した音声ファイルを保持するか
            
        Returns:
            str: 出力ファイルのパス
        """
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(f"MP4ファイルが見つかりません: {mp4_path}")
        
        audio_path = None
        try:
            # 1. 音声抽出
            audio_path = self.extract_audio_from_mp4(mp4_path)
            
            # 2. 文字起こし
            result = self.transcribe_with_timestamps(audio_path, language)
            
            # 3. 結果保存
            output_file = self.save_transcript(result, mp4_path, include_speakers, include_timestamps)
            
            return output_file
            
        finally:
            # 4. 一時音声ファイルのクリーンアップ
            if audio_path and os.path.exists(audio_path) and not keep_audio:
                try:
                    os.remove(audio_path)
                    print(f"一時音声ファイルを削除: {audio_path}")
                except Exception as e:
                    self.logger.warning(f"一時ファイル削除失敗: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="MP4動画ファイル音声文字起こしアプリ")
    parser.add_argument("mp4_file", help="処理するMP4ファイルのパス")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                       default="base", help="Whisperモデルサイズ (デフォルト: base)")
    parser.add_argument("--language", default="ja", help="言語コード (デフォルト: ja)")
    parser.add_argument("--output-dir", default="transcriptions", help="出力ディレクトリ")
    parser.add_argument("--no-speakers", action="store_true", help="話者識別を無効にする")
    parser.add_argument("--no-timestamps", action="store_true", help="タイムスタンプを無効にする")
    parser.add_argument("--keep-audio", action="store_true", help="抽出した音声ファイルを保持する")
    
    args = parser.parse_args()
    
    # アプリケーション初期化
    app = MP4TranscriptionApp(model_size=args.model, output_dir=args.output_dir)
    
    print("🎬 MP4音声文字起こしアプリ")
    print(f"📁 入力ファイル: {args.mp4_file}")
    print(f"🧠 使用モデル: {args.model}")
    print(f"🌐 言語: {args.language}")
    print("-" * 50)
    
    try:
        # MP4ファイルを処理
        output_file = app.process_mp4(
            mp4_path=args.mp4_file,
            language=args.language,
            include_speakers=not args.no_speakers,
            include_timestamps=not args.no_timestamps,
            keep_audio=args.keep_audio
        )
        
        print(f"\n🎉 処理完了！")
        print(f"📄 出力ファイル: {output_file}")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()