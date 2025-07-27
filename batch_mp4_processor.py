#!/usr/bin/env python3
"""
複数のMP4ファイルを一括で文字起こし処理するスクリプト
"""

import os
import glob
import argparse
from pathlib import Path
import time
from mp4_transcription import MP4TranscriptionApp

class BatchMP4Processor:
    def __init__(self, model_size="base", output_dir="transcriptions"):
        self.app = MP4TranscriptionApp(model_size=model_size, output_dir=output_dir)
        self.processed_files = []
        self.failed_files = []
    
    def find_mp4_files(self, directory, recursive=True):
        """
        ディレクトリからMP4ファイルを検索
        
        Args:
            directory (str): 検索ディレクトリ
            recursive (bool): サブディレクトリも検索するか
            
        Returns:
            list: MP4ファイルのパスリスト
        """
        if recursive:
            pattern = os.path.join(directory, "**", "*.mp4")
            mp4_files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(directory, "*.mp4")
            mp4_files = glob.glob(pattern)
        
        return sorted(mp4_files)
    
    def process_files(self, file_paths, language="ja", include_speakers=True, 
                     include_timestamps=True, keep_audio=False):
        """
        複数のMP4ファイルを順次処理
        
        Args:
            file_paths (list): 処理するファイルパスのリスト
            language (str): 言語コード
            include_speakers (bool): 話者識別を含めるか
            include_timestamps (bool): タイムスタンプを含めるか
            keep_audio (bool): 抽出した音声ファイルを保持するか
        """
        total_files = len(file_paths)
        
        print(f"🎬 {total_files}個のMP4ファイルを処理開始")
        print("=" * 60)
        
        start_time = time.time()
        
        for i, mp4_path in enumerate(file_paths, 1):
            print(f"\n📹 [{i}/{total_files}] 処理中: {Path(mp4_path).name}")
            print("-" * 40)
            
            file_start_time = time.time()
            
            try:
                output_file = self.app.process_mp4(
                    mp4_path=mp4_path,
                    language=language,
                    include_speakers=include_speakers,
                    include_timestamps=include_timestamps,
                    keep_audio=keep_audio
                )
                
                file_duration = time.time() - file_start_time
                self.processed_files.append((mp4_path, output_file))
                
                print(f"✅ 完了 ({file_duration:.1f}秒)")
                
            except Exception as e:
                file_duration = time.time() - file_start_time
                self.failed_files.append((mp4_path, str(e)))
                
                print(f"❌ エラー ({file_duration:.1f}秒): {e}")
        
        total_duration = time.time() - start_time
        
        # 処理結果サマリー
        self.print_summary(total_duration)
    
    def print_summary(self, total_duration):
        """処理結果のサマリーを表示"""
        print("\n" + "=" * 60)
        print("🎉 一括処理完了")
        print("=" * 60)
        
        print(f"⏱️  総処理時間: {total_duration:.1f}秒 ({total_duration/60:.1f}分)")
        print(f"✅ 成功: {len(self.processed_files)}ファイル")
        print(f"❌ 失敗: {len(self.failed_files)}ファイル")
        
        if self.processed_files:
            print("\n【成功したファイル】")
            for mp4_path, output_file in self.processed_files:
                print(f"  📁 {Path(mp4_path).name} -> {Path(output_file).name}")
        
        if self.failed_files:
            print("\n【失敗したファイル】")
            for mp4_path, error in self.failed_files:
                print(f"  ❌ {Path(mp4_path).name}: {error}")
    
    def process_directory(self, directory, **kwargs):
        """
        ディレクトリ内のすべてのMP4ファイルを処理
        
        Args:
            directory (str): 処理するディレクトリ
            **kwargs: process_filesに渡すその他の引数
        """
        mp4_files = self.find_mp4_files(directory, recursive=kwargs.get('recursive', True))
        
        if not mp4_files:
            print(f"❌ MP4ファイルが見つかりません: {directory}")
            return
        
        print(f"📁 ディレクトリ: {directory}")
        print(f"🔍 見つかったMP4ファイル: {len(mp4_files)}個")
        
        # ファイルリストを表示
        for mp4_file in mp4_files:
            print(f"  📹 {Path(mp4_file).name}")
        
        # 確認
        response = input(f"\n{len(mp4_files)}個のファイルを処理しますか？ (y/n): ").lower()
        if response != 'y':
            print("❌ 処理をキャンセルしました")
            return
        
        self.process_files(mp4_files, **kwargs)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="複数MP4ファイル一括文字起こし")
    
    # 入力方法の選択
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--directory", "-d", help="処理するディレクトリパス")
    input_group.add_argument("--files", "-f", nargs="+", help="処理するMP4ファイルパス（複数指定可能）")
    
    # オプション
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                       default="base", help="Whisperモデルサイズ")
    parser.add_argument("--language", default="ja", help="言語コード")
    parser.add_argument("--output-dir", default="transcriptions", help="出力ディレクトリ")
    parser.add_argument("--no-recursive", action="store_true", help="サブディレクトリを検索しない")
    parser.add_argument("--no-speakers", action="store_true", help="話者識別を無効にする")
    parser.add_argument("--no-timestamps", action="store_true", help="タイムスタンプを無効にする")
    parser.add_argument("--keep-audio", action="store_true", help="抽出した音声ファイルを保持する")
    
    args = parser.parse_args()
    
    # バッチプロセッサ初期化
    processor = BatchMP4Processor(model_size=args.model, output_dir=args.output_dir)
    
    print("🎬 MP4一括文字起こしアプリ")
    print(f"🧠 使用モデル: {args.model}")
    print(f"🌐 言語: {args.language}")
    
    # 処理オプション
    process_options = {
        'language': args.language,
        'include_speakers': not args.no_speakers,
        'include_timestamps': not args.no_timestamps,
        'keep_audio': args.keep_audio
    }
    
    if args.directory:
        # ディレクトリ処理
        process_options['recursive'] = not args.no_recursive
        processor.process_directory(args.directory, **process_options)
    else:
        # 個別ファイル処理
        processor.process_files(args.files, **process_options)

if __name__ == "__main__":
    main()