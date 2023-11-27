# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from sort import *
from tools.draw_tools import draw_bboxes,save_image_based_on_sub_frame,filter_detections_inside_polygon,draw_boxes_entrance_exit,find_polygons_for_centroids,calculate_direction,draw_on_frame,create_image_banner
from tools.PersonImageComparer import PersonImageComparer
from tools.PersonImage import PersonImage
import re

class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, tracker=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.tracker = tracker
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, num_frame=0):
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                original_frame = frame.copy()
                dets_to_sort = torch.cat((predictions.pred_boxes.tensor,predictions.scores.unsqueeze(1), predictions.pred_classes.unsqueeze(1)),dim=1).numpy()
                dets_to_sort = dets_to_sort[dets_to_sort[:, 5] == 0.0] #Filter only class 0
                dets_to_sort = filter_detections_inside_polygon(detections=dets_to_sort)
                # BYTE TRACK
                polygons = draw_boxes_entrance_exit(frame)
                if(len(dets_to_sort) > 0):
                    online_targets = self.tracker.update(dets_to_sort[:,:5],frame.shape,frame.shape)
                    identities = [obj.track_id for obj in online_targets]
                    scores = [obj.score for obj in online_targets]
                    bboxes = [np.concatenate((obj.tlbr,[identity],[score])) for obj, identity, score in zip(online_targets, identities,scores)]            
                    frame = draw_bboxes(frame,bboxes,num_frame=num_frame)
                    if online_targets.__len__() > 0:
                        FOLDER_PATH='in-out'
                        for target in online_targets:
                            polygons_indexes = find_polygons_for_centroids(target.history,polygons,frame,target.max_len_history)
                            if polygons_indexes is not None:
                                if polygons_indexes['direction'] is not None and polygons_indexes['between_polygons'] is not None:
                                    one_person = np.concatenate((target.tlbr, [target.track_id, target.score]))
                                    save_image_based_on_sub_frame(num_frame=num_frame,img=original_frame,boxes=[one_person],frame_step=5,directory_name=FOLDER_PATH,direction=polygons_indexes['direction'])
                        # This will run after all the persons have been processeds
                        for target in online_targets:
                            if target.history.__len__() == target.max_len_history:
                                polygons_indexes = find_polygons_for_centroids(target.history,polygons,frame,target.max_len_history)
                                if polygons_indexes is not None:
                                    direction = polygons_indexes['direction']
                                    if(direction is not None):
                                        dir_path = os.path.join(FOLDER_PATH, str(target.track_id))
                                        all_files = glob.glob(os.path.join(dir_path, '*'))
                                        image_files = [file for file in all_files if os.path.splitext(file)[1].lower() in  ['.jpg', '.jpeg', '.png']]
                                        PersonImageComparer.process_person_image(PersonImage(target.track_id,image_files , direction))
                    draw_on_frame(frame,PersonImageComparer,num_frame=num_frame,frame_step=1)
                dir_path = 'in-out/1'
                all_files = glob.glob(os.path.join(dir_path, '*'))
                def extract_id(filename):
                    match = re.search(r'img_(\d+)', filename)
                    if match:
                        return int(match.group(1))
                    else:
                        return 0
                image_files = sorted([file for file in all_files if os.path.splitext(file)[1].lower() in ['.png']],key=extract_id)
                print(image_files)
                create_image_banner(image_files, frame.shape[1], frame)
                # dir_path = 'in-out/55'
                # all_files = glob.glob(os.path.join(dir_path, '*'))
                # image_files = [file for file in all_files if os.path.splitext(file)[1].lower() in ['.png']]
                # create_image_banner(image_files, frame.shape[1], frame,100)
                
                # For Feedback purpose
                
                return frame
                vis_frame = video_visualizer.draw_instance_predictions(frame, [])
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            # vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for idx,frame in enumerate(frame_gen):
                yield process_predictions(frame, self.predictor(frame), idx+1)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
