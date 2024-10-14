import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipyw
import cc3d  
import cv2
from IPython.display import display, clear_output


class PatientViewer3D:
    """
    A class for displaying 3D medical images with masks and interactive controls.
    """
    def create_names(self, volume: np.ndarray, mask: np.ndarray | None) -> None:
        """
        Initializes volume and mask, computes connected components in the mask.
        Args:
            volume (np.ndarray): 3D medical image.
            mask (Optional[np.ndarray]): 3D segmentation mask. If None, a zero mask is used.
        """
        self.volume = volume
        self.mask = mask if mask is not None else np.zeros_like(volume)
        # Compute connected components and the number of components
        self.labels_out, self.N = cc3d.largest_k(
            self.mask.astype(np.uint8), k=30, connectivity=26, delta=0, return_N=True
        )
        self.vol, self.seg = self.volume, self.mask
    
    def __init__(
            self,
            volume: np.ndarray,
            mask: np.ndarray | None = None,
            figsize: tuple[int, int] = (7, 7),
    ) -> None:
        """
        Initializes the widget with the 3D image and mask, sets up initial controls.
        Args:
            volume (np.ndarray): 3D medical image.
            mask (Optional[np.ndarray]): Segmentation mask. Default is None.
            figsize (tuple[int, int]): The size of the figure for display.
        """
        self.create_names(volume, mask)
        self.figsize = figsize
        self.cmap = plt.cm.gray  # type: ignore
        self.widget_changed = False
        # Create and display widgets
        self.create_widgets()
        display(ipyw.HBox([self.view_widget, self.clip_widget]))
        display(ipyw.HBox([self.reverse_bbox, self.global_bbox, self.component_bbox, self.is_contour, self.alpha]))
        display(ipyw.HBox([self.clip, self.component_slicer]))
        display(ipyw.HBox([self.z]))
        self.output = ipyw.Output()
        display(self.output)
        self.update_view()
    
    def create_widgets(self) -> None:
        """
        Creates interactive widgets for controlling the 3D image display.
        """
        self.view_widget = ipyw.ToggleButtons(
            options=['axial', 'sagittal', 'coronal'],
            value='sagittal',
            description='Slice plane selection:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self.view_widget.style.button_width = '80px'
        self.view_widget.style.font_size = '10px'
        self.clip_widget = ipyw.ToggleButtons(
            options=['Soft', 'Lung', 'Brain', 'Bone'],
            value='Soft',
            description='Window preset selection:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self.clip_widget.style.button_width = '80px'
        self.clip_widget.style.font_size = '10px'
        self.reverse_bbox = ipyw.ToggleButtons(
            options=['Off', 'On'],
            value='Off',
            description='Flip on:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self.reverse_bbox.style.button_width = '80px'
        self.reverse_bbox.style.font_size = '10px'
        
        self.global_bbox = ipyw.ToggleButtons(
            options=['Off', 'On'],
            value='Off',
            description='Global bbox:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self.global_bbox.style.button_width = '80px'
        self.global_bbox.style.font_size = '10px'
        
        self.component_bbox = ipyw.ToggleButtons(
            options=['Off', 'On'],
            value='Off',
            description='Component bbox:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self.component_bbox.style.button_width = '80px'
        self.component_bbox.style.font_size = '10px'
        
        self.is_contour = ipyw.ToggleButtons(
            options=['Off', 'On'],
            value='Off',
            description='Contour:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self.is_contour.style.button_width = '80px'
        self.is_contour.style.font_size = '10px'
        
        self.alpha = ipyw.FloatSlider(
            min=0, max=1.0, step=0.01, continuous_update=True, description='Alpha:', value=0.25
        )
        self.z = ipyw.IntSlider(
            min=0, max=self.volume.shape[2] - 1, step=1,
            continuous_update=True, description='Slice:', value=self.volume.shape[2] // 2,
            layout=ipyw.Layout(width='500px', height='20px')
        )
        self.clip = ipyw.IntRangeSlider(
            value=[-160, 240], min=-2000, max=2000, step=1,
            description='Clip:', disabled=False, continuous_update=True,
            orientation='horizontal', readout=True, readout_format='d',
            layout=ipyw.Layout(width='500px', height='20px')
        )
        self.component_slicer = ipyw.SelectionSlider(
            options=['all'] + [x for x in range(1, self.N)],
            value='all',
            description='Components',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        # Add observers to handle widget changes
        self.reverse_bbox.observe(self.on_widget_change_flip, names='value')
        self.clip_widget.observe(self.on_widget_change_clip, names='value')
        self.component_slicer.observe(self.on_widget_change_comp, names='value')
        self.view_widget.observe(self.on_widget_change, names='value')
        self.global_bbox.observe(self.on_widget_change, names='value')
        self.component_bbox.observe(self.on_widget_change, names='value')
        self.is_contour.observe(self.on_widget_change, names='value')
        self.alpha.observe(self.on_widget_change, names='value')
        self.z.observe(self.on_widget_change, names='value')
        self.clip.observe(self.on_widget_change, names='value')
    
    def on_widget_change_clip(self, change: dict) -> None:
        """
        Updates the view based on the clip selection.
        Args:
            change (dict): The change information from the widget.
        """
        if self.clip_widget.value == 'Soft':
            self.clip.value = (-160, 240)
        elif self.clip_widget.value == 'Lung':
            self.clip.value = (-1000, 0)
        elif self.clip_widget.value == 'Brain':
            self.clip.value = (-40, 80)
        elif self.clip_widget.value == 'Bone':
            self.clip.value = (-500, 1300)
        self.view_selection()
        self.update_view()
        
    def on_widget_change_flip(self, change: dict) -> None:
        """
        Updates the view when a widget is changed.
        Args:
            change (dict): The change information from the widget.
        """
        self.volume = self.volume[::-1]
        self.mask = self.mask[::-1]
        self.view_selection()
        self.update_view()
    
    def on_widget_change(self, change: dict) -> None:
        """
        Updates the view when a widget is changed.
        Args:
            change (dict): The change information from the widget.
        """
        self.view_selection()
        self.update_view()
    
    def on_widget_change_comp(self, change: dict) -> None:
        """
        Updates the view based on the component selection.
        Args:
            change (dict): The change information from the component slicer widget.
        """
        self.view_selection()
        if self.component_slicer.value != 'all':
            self.z.value = np.argmax(np.sum((self.labels == self.component_slicer.value), axis=(0, 1)))
        self.update_view()
    
    def view_selection(self) -> None:
        """
        Updates the selected view (axial, sagittal, coronal) and applies it to the volume and mask.
        """
        view = self.view_widget.value
        orient = {"axial": [1, 2, 0], "sagittal": [0, 1, 2], "coronal": [0, 2, 1]}
        self.vol = np.transpose(self.volume, orient[view])
        self.seg = np.transpose(self.mask, orient[view])
        self.labels = np.transpose(self.labels_out, orient[view])

    def update_view(self) -> None:
        """
        Updates and displays the current 2D slice of the volume and mask.
        """
        z = max(0, min(self.z.value, self.vol.shape[2] - 1))
        clip_min, clip_max = self.clip.value
        global_bbox = self.global_bbox.value == 'On'
        component_bbox = self.component_bbox.value == 'On'
        is_contour = self.is_contour.value == 'On'
        alpha = self.alpha.value
        with self.output:
            clear_output(wait=True)
            plt.close()
            self.fig = plt.figure(figsize=self.figsize)
            ax = self.fig.add_subplot(111)
            # Display the volume slice and mask slice
            image = self.vol[:, :, z]
            mask = self.seg[:, :, z].astype(np.uint8)
            ax.imshow(image, cmap=self.cmap, vmin=clip_min, vmax=clip_max)
            ax.imshow(mask, alpha=alpha)
            # Add global bounding box
            if global_bbox:
                rect = self.get_rectangle(mask, edgecolor='green')
                if rect:
                    ax.add_patch(rect)
            # Add contour lines if selected
            if is_contour:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt = np.array(cnt)
                    ax.plot(cnt[:, :, 0], cnt[:, :, 1], '--r', linewidth=0.7)
            # Add component bounding boxes
            if component_bbox:
                labels_out, N = cc3d.largest_k(mask, k=30, connectivity=26, delta=0, return_N=True)
                for i in range(N):
                    rect = self.get_rectangle(labels_out == (i + 1), edgecolor='blue')
                    if rect:
                        ax.add_patch(rect)
            plt.show()
    
    def get_rectangle(self, mask: np.ndarray, edgecolor: str = 'green') -> plt.Rectangle | None:
        """
        Creates a bounding box rectangle for a given mask.
        Args:
            mask (np.ndarray): Binary mask.
            edgecolor (str): Color of the rectangle edges.
        Returns:
            Optional[plt.Rectangle]: The bounding box as a matplotlib rectangle or None.
        """
        pos = np.where(mask)
        if len(pos[0]) > 0:
            y_min, y_max = float(np.min(pos[0])), float(np.max(pos[0]))
            x_min, x_max = float(np.min(pos[1])), float(np.max(pos[1]))
            return plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1, edgecolor=edgecolor, facecolor='none'
            )
        return None