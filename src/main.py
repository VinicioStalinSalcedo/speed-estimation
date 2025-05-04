from dataset.dataset_info import ALL_VIDEOS
from dataset.session_recording import SessionRecording
from processing.process_video import process_video


if __name__ == '__main__':
    # for (session, recording) in ALL_VIDEOS:
    # for (session, recording) in [('session1', 'center')]:
    for (session, recording) in [('session0', 'right')]:
    # for (session, recording) in [('session0', 'right'),
    #                              ('session1', 'center'),
    #                              ('session2', 'center'),
    #                              ('session3', 'center'),
    #                              ('session4', 'center'),
    #                              ('session5', 'center'),
    #                              ]:
        sr = SessionRecording(session, recording)
        sr.load_data()
        process_video(sr)
