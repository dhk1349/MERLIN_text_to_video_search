import cv2
import time
from PIL import Image as Img
from sklearn.metrics.pairwise import cosine_similarity

def get_video_duration(video_path):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 프레임 수와 FPS를 통해 비디오 길이 계산
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # 리소스 해제
    cap.release()
    return duration


def video_frame_generator(video_path: str, resize_factor: float = 1.0, skip_second: int = 1):
    """
    비디오 피쳐 추출 함수
    """
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    

    resize_width = 320 # int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    resize_height = 240 # int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)

    skip_rate = int(round(video_fps) * skip_second)

    frame_number = 0
    processed_frame_count = 0

    total_grab_time = 0
    total_retrieve_time = 0

    for i in range(video_total_frame_count):
        tmp = time.time()
        success = video.grab()
        if not success:
            break  # end of the video

        frame_number += 1
        if frame_number % skip_rate == 0:
            processed_frame_count += 1

            tmp = time.time()
            status, frame = video.retrieve()
            total_retrieve_time += time.time() - tmp

            frame = cv2.resize(frame, (resize_width, resize_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Img.fromarray(frame)
            yield frame
    # print(f"Video FPS: {video_fps}, Total Frame Count: {video_total_frame_count} =>  {video_path.split('/')[-1]}, {processed_frame_count} of frames")
    video.release()


# Calculate cosine similarity
def cosine_sim(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

