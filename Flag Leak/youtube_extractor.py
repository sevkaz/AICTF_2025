import requests
from bs4 import BeautifulSoup
import re
import os
import yt_dlp
import time
from datetime import datetime
import csv
import sys

def get_and_sort_youtube_videos(
    url: str,
    output_csv_filename: str = "youtube_videos_sorted_by_date.csv",
    initial_delay: int = 30, # Initial delay between requests in seconds
    max_retries: int = 3     # Max attempts for fetching video metadata
):
    """
    Fetches YouTube video links from a given URL, retrieves video title and publish date
    using yt-dlp, sorts them from newest to oldest, and saves to a CSV file.
    Includes a retry mechanism with increasing delays to handle YouTube's rate limiting.
    """
    print(f"Fetching content from URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        youtube_links = []
        for tag in soup.find_all(['a', 'iframe']):
            src_or_href = tag.get('href') if tag.name == 'a' else tag.get('src')

            if src_or_href and ("youtube.com/watch?" in src_or_href or "youtu.be/" in src_or_href):
                video_id_match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', src_or_href)
                if video_id_match:
                    video_id = video_id_match.group(1)
                    youtube_links.append(f"https://www.youtube.com/watch?v={video_id}")

        if not youtube_links:
            print("No YouTube video links found on the page matching the specified patterns.")
            return

        print(f"Found {len(youtube_links)} YouTube video links. Fetching metadata (initial delay: {initial_delay}s)...")

        video_data = []
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'force_generic_extractor': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for i, link in enumerate(youtube_links):
                attempt = 0
                while attempt < max_retries:
                    print(f"  {i+1}/{len(youtube_links)}: Fetching metadata for '{link}' (Attempt {attempt + 1}/{max_retries})...")
                    try:
                        info = ydl.extract_info(link, download=False)
                        upload_date_str = info.get('upload_date')
                        video_title = info.get('title', 'Title Not Found')

                        if upload_date_str:
                            upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
                            video_data.append({'link': link, 'title': video_title, 'date': upload_date})
                            break # Success, move to next video
                        else:
                            print(f"    Warning: No publish date found for '{link}'. Skipping.")
                            break # No date, no need to retry
                    except yt_dlp.utils.DownloadError as e:
                        print(f"    Error fetching metadata for '{link}': {e}")
                        if "rate-limited" in str(e).lower() or "video unavailable" in str(e).lower():
                            print(f"    WARNING: Rate-limited or video unavailable. Retrying with increased delay.")
                            attempt += 1
                            time.sleep(initial_delay * (attempt + 1)) # Increase delay for subsequent retries
                        else:
                            print(f"    Error not resolvable by retry. Skipping.")
                            break # Other errors, don't retry
                    except Exception as e:
                        print(f"    An unexpected error occurred for '{link}': {e}. Skipping.")
                        break # Other errors, don't retry
                else:
                    print(f"    WARNING: Failed to fetch metadata for '{link}' after {max_retries} attempts. Skipping.")

        if not video_data:
            print("No YouTube videos with obtainable publish dates were found.")
            return

        # Sort videos by publish date, newest to oldest
        video_data.sort(key=lambda x: x['date'], reverse=True)

        # Save to CSV
        print(f"\nSaving sorted videos to '{output_csv_filename}'...")
        with open(output_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Publish Date', 'Video Title', 'URL']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item in video_data:
                writer.writerow({
                    'Publish Date': item['date'].strftime('%Y-%m-%d'),
                    'Video Title': item['title'],
                    'URL': item['link']
                })

        print(f"YouTube video information successfully saved to '{output_csv_filename}'.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}. Please check the URL and your internet connection.")
    except Exception as e:
        print(f"An unexpected general error occurred: {e}")

if __name__ == "__main__":
    target_url = "https://aictf.phdays.fun/files/flagleak_d3eec25.html"
    output_csv_file = "youtube_videos_sorted_by_date.csv"

    # Configure delay and retries here:
    custom_initial_delay_seconds = 30 # Initial delay for each video request
    custom_max_retries = 3           # Max attempts for each video

    print(f"Script starting with initial delay: {custom_initial_delay_seconds}s and max retries: {custom_max_retries}.")
    get_and_sort_youtube_videos(target_url, output_csv_file, custom_initial_delay_seconds, custom_max_retries)
