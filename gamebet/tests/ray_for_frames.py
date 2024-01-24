from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import DeveloperAPI

import ray
import cv2
import os

DATASOURCE_READER_BATCH_SIZE = 8


@DeveloperAPI
class VideoFrameDatasource(Datasource):
    def __init__(self, video_path: str):
        assert os.path.exists(video_path)
        self._video_path = video_path

    def get_read_tasks(self, parallelism):
        assert parallelism == 1
        meta = BlockMetadata(
            num_rows=None,
            size_bytes=None,
            schema=None,
            input_files=None,
            exec_stats=None,
        )
        read_task = ReadTask(
            lambda p=self._video_path: _read_frames(p),
            metadata=meta,
        )

        return [read_task]

    def estimate_inmemory_data_size(self):
        return None


def _read_frames(video_path: str):
    batch = []
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            break
        batch.append(frame)
        if len(batch) == DATASOURCE_READER_BATCH_SIZE:
            builder = DelegatingBlockBuilder()
            builder.add_batch({"frames": batch})
            yield builder.build()
            batch.clear()

    if len(batch) > 0:
        builder = DelegatingBlockBuilder()
        builder.add_batch({"frames": batch})
        yield builder.build()


if __name__ == '__main__':
    print('Test VideoFrameDatasource') 
    video_path = '/data/source/hzcsai_com/hzcsbet/gamebet/datasets/0bfacc_5.mp4'
    ds = ray.data.read_datasource(VideoFrameDatasource(video_path=video_path), parallelism=1)
    single_batch = ds.take_batch(10)
    print(dir(single_batch))
    print(ds.schema())
