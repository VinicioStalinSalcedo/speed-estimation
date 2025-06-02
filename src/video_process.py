from dataset.session_recording import SessionRecording
from processing.process_instance import process_instance_video

ALL_VIDEOS = [
    ('session1', 'center'),
    ('session1', 'left'),
    ('session1', 'right'),
    ('session2', 'center'),
    ('session2', 'left'),
    ('session2', 'right'),
    ('session3', 'center'),
    ('session3', 'left'),
    ('session3', 'right'),
    ('session4', 'center'),
    ('session4', 'left'),
    ('session4', 'right'),
    ('session5', 'center'),
    ('session5', 'left'),
    ('session5', 'right'),
    ('session6', 'center'),
    ('session6', 'left'),
    ('session6', 'right'),
]


if __name__ == '__main__':
    for (session, recording) in ALL_VIDEOS:
        sr = SessionRecording(session, recording)
        sr.load_data()
        process_instance_video(sr)
