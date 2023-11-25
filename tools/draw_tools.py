import cv2
import os
import numpy as np

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

def draw_boxes_entrance_exit(image):
    # 0 -> 1 means exit
    # 1 -> 0 means entrance
    #Inside 
    RED = (0, 0, 255) #BGR
    pts_entrance = np.array([[265, 866],[558, 683],[709, 748],[393, 998]], np.int32)
    pts_entrance = pts_entrance.reshape((-1, 1, 2))
    cv2.polylines(image, [pts_entrance], isClosed=True, color=RED, thickness=5)

    #Outside
    BLUE = (255,0,0) #BGR
    pts_exit = np.array([[194, 747],[436, 621],[558, 682],[263, 865]], np.int32)
    pts_exit = pts_exit.reshape((-1, 1, 2))
    cv2.polylines(image, [pts_exit], isClosed=True, color=BLUE, thickness=5)
    return [pts_entrance,pts_exit]

def draw_bboxes(img, bbox, offset=(0,0), num_frame=0):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2,id, _ = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        # color = COLORS_10[id%len(COLORS_10)]
        color = (255,0,0) # BGR
        label = '{:d} {:.2f}'.format(id,box[5])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def save_image_based_on_sub_frame(num_frame, img,boxes, frame_step=10,directory_name='images_subframe',direction=None):
    if num_frame % frame_step == 0:
        for i,box in enumerate(boxes):    
            x1,y1,x2,y2,id,_ = [int(i) for i in box]
            sub_frame = img[y1:y2,x1:x2].copy()

            # Convert BGR to RGB
            # sub_frame_rgb = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2RGB)
            
            id_directory = os.path.join(f"{directory_name}", str(id))
            if not os.path.exists(id_directory):
                os.makedirs(id_directory)
            simple_image_name = f"img_{id}_{num_frame}"
            if direction is not None:
                simple_image_name = f"img_{id}_{num_frame}_{direction}"
            image_name = f"{simple_image_name}_{x1}_{y1}_{x2}_{y2}_{box[5]:.2f}.png"
            save_path = os.path.join(id_directory, image_name)
            
            # Save the RGB image
            try:
                cv2.imwrite(save_path, sub_frame)
            except Exception as e:
                print(f"Error encountered: {e}")

def filter_detections_inside_polygon(detections,polygon_pts=np.array([[0,1080],[0,600],[510,500],[593,523],[603,635],[632,653],[932,535],[978,621],[756,918],[764,1080]], np.int32)):
    """
    Filters detections based on whether the midpoint of the bottom edge of their bounding box
    is inside a specified polygon.

    :param detections: A numpy array of detections with shape (N, 6), where the first four columns
                       represent the bounding box coordinates (x1, y1, x2, y2).
    :param polygon_pts: A numpy array of points defining the polygon, shape (M, 2).
    :return: A numpy array of filtered detections.
    """

    # Convert polygon points to the required shape for cv2.polygonTest
    polygon = polygon_pts.reshape((-1, 1, 2))

    # Function to check if a point is inside the polygon
    def is_point_inside_polygon(point, poly):
        return cv2.pointPolygonTest(poly, point, False) >= 0

    # Filter detections
    filtered_detections = []
    for det in detections:
        # Calculate the midpoint at the bottom of the bounding box
        midpoint_x = (det[0] + det[2]) / 2
        midpoint_y = det[3]
        midpoint = (midpoint_x, midpoint_y)

        # Check if the midpoint is inside the polygon
        if is_point_inside_polygon(midpoint, polygon):
            filtered_detections.append(det)

    return np.array(filtered_detections)


def calculate_centroid(tlwh):
    x, y, w, h = tlwh
    return np.array([x + w / 2, y + h])

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

def find_polygons_for_centroids(history_deque, polygons, frame, max_len_history):
    if len(history_deque) < 2:
        return None
    centroids = [calculate_centroid(bbox) for bbox in history_deque]
    polygon_indices = []

    for centroid in centroids:
        found_in_polygon = False
        for i, polygon in enumerate(polygons):
            if is_point_in_polygon(centroid, polygon):
                cv2.circle(frame, tuple(centroid.astype(int)), 4, COLORS_10[i], -1)
                polygon_indices.append(i)
                found_in_polygon = True
                # break

    if not found_in_polygon:
        return None
    cv2.arrowedLine(frame, tuple(centroids[0].astype(int)), tuple(centroids[-1].astype(int)), (255, 0, 0), 2, tipLength=0.3)

    return {
        'polygon_indices': polygon_indices,
        'direction': calculate_direction(polygon_indices),
        'between_polygons': between_polygons(polygon_indices),
    }

def calculate_direction(polygon_indices):
    """
    This function takes an array of 1s and 0s and returns a string representing
    the point where the transition from 1 to 0 or 0 to 1 happens.
    """
    if(polygon_indices.__len__() < 2):
        return None
    for i in range(len(polygon_indices) - 1):
        # Check if there is a transition from 1 to 0 or 0 to 1
        if polygon_indices[i] != polygon_indices[i + 1]:
            # Return '10' if the transition is from 1 to 0
            if polygon_indices[i] == 1:
                return 'Entrance'
            # Return '01' if the transition is from 0 to 1
            else:
                return 'Exit'
    # Return 'No transition found' if there is no transition in the array
    return None


def between_polygons(arr):
    if(arr.__len__() < 2):
        return None
    # Calculate middle index
    middle_index = len(arr) // 2
    
    # Adjust for even-length arrays, if needed
    if len(arr) % 2 == 0:
        middle_index -= 1
    
    # Determine start and end indices for the subarray
    start = max(0, middle_index - 4)
    end = min(len(arr), middle_index + 5)
    
    # Extract the subarray
    subarray = arr[start:end]
    
    # Count the number of 1s and 0s
    count_ones = subarray.count(1)
    count_zeros = subarray.count(0)
    
    # Check if they are equal
    return count_ones == count_zeros