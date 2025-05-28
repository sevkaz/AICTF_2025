import os
import re
import cv2
import yt_dlp
from datetime import datetime
from PIL import Image
from llava.eval.run_llava import load_model_and_tokenizer, chat
from transformers import CLIPImageProcessor

def read_video_urls(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_video_metadata(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            upload_date = info.get("upload_date")
            if upload_date:
                upload_datetime = datetime.strptime(upload_date, "%Y%m%d")
                return {
                    "url": url,
                    "upload_date": upload_datetime,
                    "id": info.get("id"),
                    "title": info.get("title")
                }
    except Exception as e:
        print(f"Metadata error: {e}")
    return None

def extract_frames_every_n_seconds(video_path, interval=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video file: {video_path}")
        return frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    for sec in range(0, int(duration), interval):
        frame_num = int(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            frame_path = f"temp_frame_{sec}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append((sec, frame_path))
    cap.release()
    return frames

def vlm_flag_check(image_path, model, tokenizer, processor, prompt):
    image = Image.open(image_path).convert("RGB")
    response = chat(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        image=image,
        query=prompt
    )
    if "FLAG_" in response:
        match = re.search(r"FLAG_[A-Za-z0-9_]+", response)
        if match:
            return match.group()
    return None

def scan_video(video, model, tokenizer, processor, prompt):
    print(f"\n📥 Downloading: {video['url']}")
    ydl_opts = {
        'quiet': True,
        'format': 'mp4[height<=360]',
        'outtmpl': f"{video['id']}.%(ext)s",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video["url"]])

        video_path = f"{video['id']}.mp4"
        frames = extract_frames_every_n_seconds(video_path, interval=1)

        for sec, frame_path in frames:
            print(f"\n🖼️ Checking frame at {sec}s...")
            flag = vlm_flag_check(frame_path, model, tokenizer, processor, prompt)
            os.remove(frame_path)
            if flag:
                print(f"\n✅ FLAG FOUND: {flag} at {sec}s in {video['url']}")
                return flag

        os.remove(video_path)
    except Exception as e:
        print(f"Video scan error: {e}")
    return None

def main():
    prompt = (
        "This is a frame from a Capture The Flag (CTF) video. "
        "If you see any text that starts with FLAG_, extract and return it exactly. "
        "Otherwise, say 'No flag found'."
    )

    print("🚀 Loading LLaVA model...")
    model_name = "llava-hf/llava-1.5-7b-hf"
    model, tokenizer, processor = load_model_and_tokenizer(model_path=model_name)

    urls = read_video_urls("deneme.txt")
    videos = [get_video_metadata(url) for url in urls if get_video_metadata(url)]
    videos.sort(key=lambda x: x["upload_date"], reverse=True)

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Scanning: {video['title']} ({video['upload_date'].strftime('%Y-%m-%d')})")
        flag = scan_video(video, model, tokenizer, processor, prompt)
        if flag:
            break
    else:
        print("\n🚫 No FLAG_ found in any video.")

if __name__ == "__main__":
    main()
