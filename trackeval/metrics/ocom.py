import numpy as np
from ._base_metric import _BaseMetric
from .. import _timing


class OCOM(_BaseMetric):

    def __init__(self, config=None):
        super().__init__()
        self.plottable = True
        self.array_labels = []
        self.float_array_fields = []
        self.integer_fields = []
        self.float_fields = ['OCOM']
        self.fields = self.float_fields
        self.summary_fields = self.float_fields

        self.use_occ_bbox = config["BBOX_BASED_OCC"]
        self.use_y_coord = config["Y_COORD_OCC"]

        if self.use_occ_bbox:
            self.checkVal = 0.
        else:   
            self.checkVal = 1.

    @_timing.time
    def eval_sequence(self, data):

        # Initialise results
        res = {}
        for field in self.float_fields:
            res[field] = 0

        OCOM_res = self.getOCOM(data, y_coord_depth=self.use_y_coord)
        res.update(OCOM_res)

        return res 

    def getOCOM(self, data, y_coord_depth):
        
        # Get mean occlusion per track - either based on bounding box intersection or from annotated visibility.
        if self.use_occ_bbox:
            data["id_visibility"] = self.getCoarseOcclusionBbox(data)
            occ_dict = self.getMeanOcclusionBboxIntersection(data, y_coord_depth)
        else:
            occ_dict = self.getMeanOcclusion(data)

        # Get global occlusion based on average of mean track occlusions
        OCOM = 0
        for id in occ_dict:
            OCOM += occ_dict[str(id)]

        if len(occ_dict) > 0:
            OCOM = OCOM/len(occ_dict)
        else:
            OCOM = 0

        res = {"OCOM": OCOM}
        return res


    def getMeanOcclusion(self, data):
        """
        Calculates the mean occlusion during occlusions for each object

        Return:
            occ_dict: a dictionary with the mean occlusion level for each object
        """

        occ_dict = {str(target): 0 for target in data["unique_ids"]}

        for target in data["unique_ids"]:
            # Find the mean occlusion level for each object
            vis_values = np.asarray(data["id_visibility"][str(target)])
            if len(vis_values) == 0:
                occ_dict.pop(str(target), None)
                continue

            mean_vis = 1-np.mean(vis_values) # 1- in order to produce the occlusion level instead of visibility
            occ_dict[str(target)] = float(mean_vis)

        return occ_dict

    
    def getCoarseOcclusionBbox(self, data):
        """
        Calculates binary object visibility based on bounding box intersections.

        The intersection is measured per subject as the ratio between the intersecting bboxes and the bbox of the respective subject.
        If a subject's bounding box intersects with any other bounding boxes, it is assigned a binary value indicating it is a part of an occlusion

        Returns:
            id_visibility: dict containing binary visibility scores per subject per frame
        """

        frames = [idx for idx in range(data["num_timesteps"])]

        id_visibility = {str(gt_id): np.asarray([self.checkVal]*len(data["id_dets"][str(gt_id)])) for gt_id in data["unique_ids"]}

        for f in frames:
            f_ids = data["gt_ids"][f]
            f_dets = data["gt_dets"][f]

            # Ensure that there are at least two subjects involved in every occlusion event
            if len(f_ids) < 2:
                continue

            # Find the intersecting regions
            for idx1 in range(len(f_ids)):
                id1_dets = f_dets[idx1]
                for idx2 in range(idx1+1, len(f_ids)):
                    id2_dets = f_dets[idx2]

                    max_top = int(max(id1_dets[1], id2_dets[1]))
                    min_bot = int(min(id1_dets[1]+id1_dets[3],
                                        id2_dets[1]+id2_dets[3]))
                    max_left = int(max(id1_dets[0], id2_dets[0]))
                    min_right = int(min(id1_dets[0]+id1_dets[2],
                                        id2_dets[0]+id2_dets[2]))

                    intersection_height = min_bot - max_top + 1
                    intersection_width = min_right - max_left + 1

                    intersection_area = max(0, intersection_width) * max(0, intersection_height)

                    if intersection_area > 0:
                        id_visibility[str(f_ids[idx1])][(f+1) == np.asarray(data["id_frames"][str(f_ids[idx1])])] = 1-self.checkVal
                        id_visibility[str(f_ids[idx2])][(f+1) == np.asarray(data["id_frames"][str(f_ids[idx2])])] = 1-self.checkVal

        return id_visibility


    def getMeanOcclusionBboxIntersection(self, data, y_coord_depth, only_nonzero_occ = True):
        """
        Calculates intersection between bounding boxes.

        The intersection is measured per subject as the ratio between the intersecting bboxes and the bbox of the respective subject.
        A boolean mask with the same size as the sequence image is constructed and the pixels in every intersecting region between bounding boxes are set to 1.
        Subsequently, a look-up is made for the bbox of every subject in order to determine how large a region of the bbox is intersected.
        Lastly, the mean occlusion is calcualted per track 

        Returns:
            occ_dict: dict containing the mean occlusion per track.
        """
        occ_intersection_dict = {str(target):[] for target in data["unique_ids"]}

        frames = [idx for idx in range(data["num_timesteps"])]

        for f in frames:
            f_ids = data["gt_ids"][f]
            f_dets = data["gt_dets"][f]

            # Ensure that there are at least two subjects involved in every occlusion event
            if len(f_ids) < 2:
                continue

            # Create a mask to contain all areas with intersecting bboxes in the given frame
            mask = np.zeros((data["image_size"][1], data["image_size"][0]), dtype=np.bool)

            # Find the intersecting regions
            for idx1, target1 in enumerate(f_ids):
                mask = np.zeros((data["image_size"][1], data["image_size"][0]), dtype=np.bool)
     
                # Save Bbox info for target1
                id1_dets = f_dets[idx1]
                top = int(id1_dets[1])
                left = int(id1_dets[0])
                height = int(id1_dets[3])
                width = int(id1_dets[2])
                for idx2, target2 in enumerate(f_ids):

                    id2_dets = f_dets[idx2]

                    # Determine intersection between bounding boxes
                    max_top = int(max(top, id2_dets[1]))
                    min_bot = int(min(top+height,
                                        id2_dets[1]+id2_dets[3]))
                    max_left = int(max(left, id2_dets[0]))
                    min_right = int(min(left+width,
                                        id2_dets[0]+id2_dets[2]))

                    # If using pseudo depth assumption based on y-coordinate, and the target1 bounding box extends further down the image, skip. 
                    if top+height == min_bot and y_coord_depth:
                        continue
                        
                    # Calculate area of intersecting bounding box
                    intersection_height = min_bot - max_top + 1
                    intersection_width = min_right - max_left + 1

                    intersection_area = max(0, intersection_width) * max(0, intersection_height)

                    # If no zero intersection between bounding boxes, skip.
                    if intersection_area == 0:
                        continue

                    # Occlusion mask
                    mask[max_top:min_bot+1, max_left:min_right+1] = True

                # Ratio of pixels in target1 bounding box compared to marked points in the occlusion mask
                id_intersection = np.sum(mask[top:top+height+1,left:left+width+1])/((height+1)*(width+1))
                occ_intersection_dict[str(target1)].append(id_intersection)

        occ_dict = {}
        for target in data["unique_ids"]:
            if len(occ_intersection_dict[str(target)]) > 0:
                occ_dict[str(target)] = float(np.mean(occ_intersection_dict[str(target)]))

        return occ_dict