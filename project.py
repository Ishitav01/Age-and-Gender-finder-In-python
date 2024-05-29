import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gensim.summarization import summarize
from nltk.tokenize import sent_tokenize

# Set up YouTube Data API credentials
API_KEY = 'AIzaSyC1g2WvdXZUC1MjcFSyB7XvMd6KJ6enG1A'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Set up Video ID of the YouTube video
VIDEO_ID = 'https://www.youtube.com/watch?v=qWdyhFiyH0Y&ab_channel=GuidingTech'

# Retrieve YouTube transcript
def get_youtube_transcript(video_id):
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        captions = youtube.captions().list(part='snippet', videoId=video_id).execute()
        caption_list = captions['items']

        transcript = ''
        for caption in caption_list:
            caption_id = caption['id']
            caption_response = requests.get(f'http://video.google.com/timedtext?lang=en&v={video_id}&id={caption_id}')
            if caption_response.status_code == 200:
                transcript += caption_response.text

        return transcript
    except HttpError as e:
        print('An HTTP error occurred:')
        print(e)

# Clean and preprocess the transcript
def preprocess_transcript(transcript):
    # Add your preprocessing steps here (e.g., removing timestamps, speaker names, punctuation, etc.)
    cleaned_transcript = transcript
    return cleaned_transcript

# Generate summary using gensim's summarize function
def generate_summary(transcript):
    # Tokenize transcript into sentences
    sentences = sent_tokenize(transcript)

    # Join sentences into a single string
    transcript_text = ' '.join(sentences)

    # Generate summary using gensim's summarize function
    summary = summarize(transcript_text)

    return summary

# Main function
def main():
    transcript = get_youtube_transcript(VIDEO_ID)
    cleaned_transcript = preprocess_transcript(transcript)
    summary = generate_summary(cleaned_transcript)
    print(summary)

if __name__ == '__main__':
    main()
