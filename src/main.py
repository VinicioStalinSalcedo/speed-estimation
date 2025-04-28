from dataset.dataset_info import ALL_VIDEOS
from dataset.session_recording import SessionRecording
from processing.process_video import process_video


if __name__ == '__main__':
    for (session, recording) in ALL_VIDEOS:
        sr = SessionRecording(session, recording)
        sr.load_data()
        process_video(sr)
