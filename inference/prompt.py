import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

from mindspore import Tensor
from PIL import Image
from mindspore import ops

class FastSAMPrompt:

    def __init__(self, image, results):
        self.results = results
        self.img = image

    def _format_results(self, result, filter=0):
        annotations = []
        n = len(result['segmentation'])
        for i in range(n):
            annotation = {}
            mask = result['segmentation'][i] == 1.0

            # if ops.sum(mask) < filter:
            #     continue
            annotation['id'] = i
            annotation['segmentation'] = mask
            annotation['bbox'] = result['bbox'][i]
            annotation['score'] = result["score"][i]
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
        return annotations
    
    def point_prompt(self, points, pointlabel):  # numpy 
        if self.results == None:
            return []
        masks = self._format_results(self.results, 0)
        target_height = self.img.shape[0]
        target_width = self.img.shape[1]
        h = masks[0]['segmentation'].shape[0]
        w = masks[0]['segmentation'].shape[1]
        if h != target_height or w != target_width:
            points = [[int(point[0] * w / target_width), int(point[1] * h / target_height)] for point in points]
        onemask = np.zeros((h, w))
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        for i, annotation in enumerate(masks):
            if type(annotation) == dict:
                mask = annotation['segmentation']
            else:
                mask = annotation
            for i, point in enumerate(points):
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                    onemask[mask] = 1
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                    onemask[mask] = 0
        onemask = onemask >= 1
        return np.array([onemask])
    
    def box_prompt(self, bbox=None, bboxes=None):
        if self.results == None:
            return []
        assert bbox or bboxes
        if bboxes is None:
            bboxes = [bbox]
        max_iou_index = []
        for bbox in bboxes:
            assert (bbox[2] != 0 and bbox[3] != 0)
            masks = self.results['segmentation']
            masks = Tensor(masks)
            target_height = self.img.shape[0]
            target_width = self.img.shape[1]
            h = masks.shape[1]
            w = masks.shape[2]
            if h != target_height or w != target_width:
                bbox = [
                    int(bbox[0] * w / target_width),
                    int(bbox[1] * h / target_height),
                    int(bbox[2] * w / target_width),
                    int(bbox[3] * h / target_height), ]
            bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
            bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
            bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
            bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

            # IoUs = torch.zeros(len(masks), dtype=torch.float32)
            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            masks_area = ops.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
            orig_masks_area = ops.sum(masks, dim=(1, 2))
            union = bbox_area + orig_masks_area - masks_area
            IoUs = masks_area / union
            max_iou_index.append(int(ops.argmax(IoUs)))
        max_iou_index = list(set(max_iou_index))
        return np.array(masks[max_iou_index].numpy())
    
    def everything_prompt(self):
        if self.results == None:
            return []
        return self.results['segmentation']
    
    def plot_to_result(self,
            annotations,
            bboxes=None,
            points=None,
            point_label=None,
            mask_random_color=True,
            better_quality=True,
            retina=False,
            withContours=True) -> np.ndarray:
        if isinstance(annotations[0], dict):
            annotations = [annotation['segmentation'] for annotation in annotations]
        image = self.img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h = image.shape[0]
        original_w = image.shape[1]
        if sys.platform == "darwin":
            plt.switch_backend("TkAgg")
        plt.figure(figsize=(original_w / 100, original_h / 100))
        # Add subplot with no margin.
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(image)
        if better_quality:
            for i, mask in enumerate(annotations):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        
        annotations = np.array(annotations)
        self.fast_show_mask(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bboxes=bboxes,
            points=points,
            pointlabel=point_label,
            retinamask=retina,
            target_height=original_h,
            target_width=original_w,
        )
        if withContours:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                if type(mask) == dict:
                    mask = mask['segmentation']
                annotation = mask.astype(np.uint8)
                if not retina:
                    annotation = cv2.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_all.append(contour)
            cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
            color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
            plt.imshow(contour_mask)

        plt.axis('off')
        fig = plt.gcf()
        plt.draw()

        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        plt.close()
        return result
    
    def plot(self,
             annotations,
             output_path,
             bboxes=None,
             points=None,
             point_label=None,
             mask_random_color=True,
             better_quality=True,
             retina=False,
             withContours=True):
        if len(annotations) == 0:
            return None
        result = self.plot_to_result(
            annotations, 
            bboxes, 
            points, 
            point_label, 
            mask_random_color,
            better_quality, 
            retina, 
            withContours,
        )
        path = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(path):
            os.makedirs(path)
        result = result[:, :, ::-1]
        cv2.imwrite(output_path, result)
        return result

    def fast_show_mask(
        self,
        annotation,
        ax,
        random_color=False,
        bboxes=None,
        points=None,
        pointlabel=None,
        retinamask=True,
        target_height=960,
        target_width=960,
    ):
        msak_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]

        #Sort annotations based on area.
        areas = np.sum(annotation, axis=(1, 2))
        sorted_indices = np.argsort(areas)
        annotation = annotation[sorted_indices]
        index = (annotation != 0).argmax(axis=0)
        if random_color:
            color = np.random.random((msak_sum, 1, 1, 3))
        else:
            color = np.ones((msak_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
        transparency = np.ones((msak_sum, 1, 1, 1)) * 0.6
        visual = np.concatenate([color, transparency], axis=-1)
        mask_image = np.expand_dims(annotation, -1) * visual
        show = np.zeros((height, weight, 4))
        h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        # Use vectorized indexing to update the values of 'show'.
        show[h_indices, w_indices, :] = mask_image[indices]
        if bboxes is not None:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
        # draw point
        if points is not None:
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
                s=20,
                c='y',
            )
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
                s=20,
                c='m',
            )

        if not retinamask:
            show = cv2.resize(show, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        ax.imshow(show)