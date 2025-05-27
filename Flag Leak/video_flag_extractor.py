import os
import re
import cv2
import pytesseract
import yt_dlp
from datetime import datetime

def read_video_urls(txt_path):
    """
    Reads video URLs from a given text file.
    Each line in the file is considered a separate URL.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_video_metadata(url):
    """
    Retrieves metadata for a given YouTube video URL using yt-dlp.
    Returns a dictionary containing 'url', 'upload_date', 'id', and 'title' if successful,
    otherwise returns None.
    """
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
        print(f"⚠️ Metadata error: {e}")
    return None

def extract_frames_every_n_seconds(video_path, interval=1):
    """
    Extracts frames from a video at a specified interval (in seconds).
    Returns a list of tuples, where each tuple contains (second, frame_image).
    Handles cases where FPS or total frames cannot be read.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # If FPS is 0 or unreadable, use a default value (e.g., 30)
    if fps == 0:
        print(f"⚠️ Warning: Video FPS value could not be read or is 0. Defaulting to FPS=30: {video_path}")
        fps = 30 # Or an appropriate default value

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # If total_frames or fps is 0, duration calculation might be problematic.
    if total_frames == 0 or fps == 0:
        print(f"⚠️ Warning: Video duration cannot be calculated as total frames or FPS is zero: {video_path}")
        cap.release()
        return frames # Return an empty list of frames

    duration = total_frames / fps
    for sec in range(0, int(duration), interval):
        frame_num = int(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            frames.append((sec, frame))
    cap.release()
    return frames

def detect_flag_with_tesseract(frames, video_id):
    """
    Detects a 'FLAG_' pattern in video frames using Tesseract OCR.
    Saves extracted frames and OCR output to a directory named 'frames_{video_id}'.
    Applies image preprocessing (resizing, grayscale, adaptive thresholding) for better OCR.
    Returns the found flag and its source (visual at {second} sec) if detected, otherwise None.
    """
    output_dir = f"frames_{video_id}"
    os.makedirs(output_dir, exist_ok=True)

    for sec, frame in frames:
        try:
            # Save the frame as .jpg
            frame_path = f"{output_dir}/frame_{sec}.jpg"
            cv2.imwrite(frame_path, frame)

            # Magnification for OCR (for better OCR results)
            scale_percent = 300 # You can adjust this value as needed
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            # Preprocessing for OCR
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2 # This block size and C value can also affect OCR results
            )

            # OCR configuration
            config = "--psm 6" # Page segmentation mode, can be changed based on text layout
            text = pytesseract.image_to_string(thresh, config=config)

            # Save OCR output
            with open(f"{output_dir}/ocr_{sec}.txt", "w", encoding="utf-8") as f:
                f.write(text)

            # Normalization
            normalized = re.sub(r"[^A-Z0-9_]", "", text.upper())

            print(f"\n🔎 Frame {sec}s")
            print(f"📝 Normalized: {normalized}")

            # FLAG_ match
            match = re.search(r"FLAG_[A-Z0-9_]{4,}", normalized)
            if match:
                print(f"✅ Match found: {match.group()}")
                return match.group(), f"visual at {sec} sec"

        except Exception as e:
            print(f"⚠️ OCR error at {sec}s: {e}")
    return None, None

def scan_description_and_subs(info, video_id):
    """
    Scans the video description and downloaded subtitles for a 'FLAG_' pattern.
    Returns the found flag and its source ('description' or 'subtitles') if detected, otherwise None.
    """
    description = info.get("description", "")
    match = re.search(r"FLAG_[A-Za-z0-9_]+", description)
    if match:
        return match.group(), "description"

    # Possible subtitle file names based on yt-dlp's common naming conventions
    possible_sub_files = [
        f"{video_id}.en.vtt",
        f"{info.get('id')}.en.vtt", # Sometimes the ID in info might differ from the actual file name
        # If automatic subtitles are downloaded with a different name (e.g., videoID.en.auto.vtt), add it here
    ]

    found_subtitle_file = None
    for sub_file_candidate in possible_sub_files:
        if os.path.exists(sub_file_candidate):
            found_subtitle_file = sub_file_candidate
            break

    if found_subtitle_file:
        with open(found_subtitle_file, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"FLAG_[A-Za-z0-9_]+", content)
            if match:
                # You can optionally delete the subtitle file if no longer needed
                # try:
                #     os.remove(found_subtitle_file)
                # except OSError as e:
                #     print(f"⚠️ Subtitle file could not be deleted: {e}")
                return match.group(), "subtitles"
    return None, None

def scan_video(video):
    """
    Downloads a video, then scans its description, subtitles, and frames for a 'FLAG_' pattern.
    Prioritizes description/subtitles scan before frame extraction.
    Deletes the downloaded video file after processing.
    """
    print(f"\n📥 Downloading: {video['url']}")
    ydl_opts = {
        'quiet': True,
        # REVISED FORMAT OPTION:
        # Selects the best available MP4 video format with at least 720p height,
        # if not available, selects the best available MP4 format.
        # Higher resolution is targeted to improve OCR quality.
        'format': 'bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height>=720][ext=mp4]/best[ext=mp4]',
        # Alternatively, a simpler format for video only (without audio):
        # 'format': 'bestvideo[height>=720][ext=mp4]/best[ext=mp4]',
        # Or you can target 1080p (larger files, longer download):
        # 'format': 'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[height>=1080][ext=mp4]/best[ext=mp4]',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'outtmpl': f"{video['id']}.%(ext)s", # Ensures the downloaded file name starts with the video ID
        'postprocessors': [{ # To merge if video and audio are downloaded separately
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4', # Ensures the final output is mp4
        }],
    }

    downloaded_video_path = None
    info_dict = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video["url"], download=True)
            # It might be more reliable to get the downloaded file name from info_dict
            # yt-dlp can sometimes use different file names than specified in 'outtmpl' (e.g., _video_id)
            # However, 'outtmpl' should generally match video['id'].
            downloaded_video_path = f"{video['id']}.mp4" # or info_dict.get('requested_downloads')[0]['filepath']

            # If there's a '_filename' or 'filepath' key in info_dict, it's more accurate to use that
            # For example: downloaded_video_path = info_dict.get('_filename') or info_dict.get('filepath')
            # But in its simplest form, we rely on the outtmpl template:
            if not os.path.exists(downloaded_video_path) and info_dict:
                 # If the initial guess fails, try to find the file path from info_dict
                if 'requested_downloads' in info_dict and len(info_dict['requested_downloads']) > 0:
                    downloaded_video_path = info_dict['requested_downloads'][0]['filepath']
                elif '_filename' in info_dict: # Sometimes this key provides the full path
                    downloaded_video_path = info_dict['_filename']
                elif 'filepath' in info_dict: # Alternative
                     downloaded_video_path = info_dict['filepath']
                # If still not found and not video['id'] + ".mp4", there's an issue.

        if not info_dict:
            print(f"❌ Video information could not be retrieved: {video['url']}")
            return None, None

        # 1. Check description/subtitles
        # We pass info_dict to the scan_description_and_subs function
        flag, source = scan_description_and_subs(info_dict, video["id"])
        if flag:
            # Delete the video file (even if found only from description/subtitles)
            if downloaded_video_path and os.path.exists(downloaded_video_path):
                try:
                    os.remove(downloaded_video_path)
                except OSError as e:
                    print(f"⚠️ Video file could not be deleted: {e}")
            return flag, source

        # 2. Scan video frames
        if downloaded_video_path and os.path.exists(downloaded_video_path):
            print(f"🎞️ Extracting frames: {downloaded_video_path}")
            frames = extract_frames_every_n_seconds(downloaded_video_path, interval=1)
            if frames:
                flag, source = detect_flag_with_tesseract(frames, video["id"])
            # Delete the downloaded video file after processing
            try:
                os.remove(downloaded_video_path)
                print(f"🗑️ Video deleted: {downloaded_video_path}")
            except OSError as e:
                print(f"⚠️ Video file could not be deleted: {e}")

            # If flag was found in frames, return it
            if flag:
                return flag, source
        else:
            print(f"❌ Downloaded video file not found: {downloaded_video_path if downloaded_video_path else video['id']+'.mp4'}")

    except yt_dlp.utils.DownloadError as de:
        print(f"❌ Video download error (yt-dlp): {de}")
    except Exception as e:
        print(f"❌ General error during video scan: {e}")
        # Even in case of error, we can try to delete the downloaded file
        if downloaded_video_path and os.path.exists(downloaded_video_path):
            try:
                os.remove(downloaded_video_path)
            except OSError as err_remove:
                print(f"⚠️ Video file could not be deleted after error: {err_remove}")
    return None, None

def main():
    """
    Main function to orchestrate the video scanning process.
    Reads URLs from 'deneme.txt', retrieves metadata, sorts videos by upload date (newest first),
    and then scans each video for a flag.
    """
    urls = read_video_urls("video_urls.txt")
    print(f"📄 Number of video URLs loaded: {len(urls)}.")

    videos = []
    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] Getting metadata...")
        meta = get_video_metadata(url)
        if meta:
            videos.append(meta)

    # Sort videos from newest to oldest
    videos.sort(key=lambda x: x["upload_date"], reverse=True)

    found_any_flag = False
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Scanning: {video['title']} ({video['upload_date'].strftime('%Y-%m-%d')})")
        flag, source = scan_video(video)
        if flag:
            print(f"\n✅ FLAG FOUND: {flag} (source: {source})")
            print(f"🔗 Video URL: {video['url']}")
            found_any_flag = True
            # Optional: Uncomment 'break' to stop after the first flag is found,
            # otherwise all videos will be scanned.
            # break

    if not found_any_flag:
        print("\n🚫 No FLAG_ found in any video.")

if __name__ == "__main__":
    # You might need to specify the path where Pytesseract is installed, especially on Windows:
    # Example:
    # if os.name == 'nt': # For Windows
    #     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    main()
