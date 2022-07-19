import numpy as np
from ._base_metric import _BaseMetric
from .. import _timing


class MCOM(_BaseMetric):

    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        self.array_labels = []
        self.float_array_fields = []
        self.integer_fields = []
        self.float_fields = ["MCOM"]
        self.fields = self.float_fields
        self.summary_fields = self.float_fields

    @_timing.time
    def eval_sequence(self, data):

        # Initialise results
        res = {}
        for field in self.float_fields:
            res[field] = 0

        motmx_res = self.getMOTMX(data)
        res.update(motmx_res)

        return res 

    def getMOTMX(self, data):

        res = {}
        mcom_alpha = []
        
        # Alpha range values
        lowlim = 0.01
        uplim = 1.0
        step_size = 0.01

        # Get Area compensated displacement.
        bbCenters_id, areas_id = self.getBBInfo(data)
        estPosError_id = self.getEstPosError(bbCenters_id)
        areas = []
        estPostError = []
        for target in areas_id.keys():
            areas.extend(areas_id[str(target)])
            estPostError.extend(estPosError_id[str(target)])

        area_displacement = np.mean(np.asarray(estPostError)/np.sqrt(np.asarray(areas)))

        alphas = [idx for idx in np.arange(lowlim, uplim+step_size, step_size)]
        for alpha in alphas:
            mcom_alpha.append(self.log_sigmoid(area_displacement, alpha))
        
        res["MCOM"] = np.mean(mcom_alpha)
        return res

    def getBBInfo(self, data):
        centers = {str(target): [] for target in data["unique_ids"]}
        areas = {str(target): [] for target in data["unique_ids"]}
        
        for target in data["unique_ids"]:
            target_len = len(data["id_frames"][str(target)])
            for f_idx, frame in enumerate(data["id_frames"][str(target)]):

                if f_idx+1 < target_len:
                    bbox = data["id_dets"][str(target)][f_idx+1]
                else:
                    bbox = data["id_dets"][str(target)][f_idx]

                x = bbox[0] + bbox[2]/2
                y = bbox[1] + bbox[3]/2
                area = bbox[2] * bbox[3]

                centers[str(target)].append((x,y))
                areas[str(target)].append(area)
            if len(centers[str(target)]) == 0:
                centers.pop(str(target), None)
            if len(areas[str(target)]) == 0:
                areas.pop(str(target), None)

        return centers, areas

    def getEstPosError(self, centers):
        """
        Calculate the estimated positional error
        """
        errs = {str(target): [] for target in centers.keys()}
        for target in centers.keys():
            center_id = centers[str(target)]
            x = [x for x,y in center_id]
            y = [y for x,y in center_id]
            pos = np.array((x,y)).transpose() # Switches from frames x coords to coords x frames

            # Find position displacement vectors from frame i to frame i+1
            gtV = pos[:-1]-pos[1:]
            # insert a zero-vector as the starting point
            gtV = np.insert(gtV,0,[0,0],axis=0)

            # Estimated vector, shifted previous displacement error
            estV = gtV[:-1]
            estV = np.insert(estV,0,[0,0],axis=0)

            # Positional error, using constant velocity motion model looking one frame behind
            err = np.linalg.norm(gtV-estV,axis=1)
            errs[str(target)].extend(list(err))

        return errs

    def log_sigmoid(self,x,alpha=10):
        return 1/(1+(alpha/x))