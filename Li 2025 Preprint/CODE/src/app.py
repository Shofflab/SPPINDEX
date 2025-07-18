import pickle as pkl
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib import colors
from matplotlib.path import Path as PltPath
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QRegExpValidator
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QWidget,
)
from scipy import ndimage, stats

# Global variable for pyqtgraph - set in main()
pg = None


class InputDialogIntensity(QWidget):
    """Dialog for collecting user inputs for intensity analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dict = {}
        self.le_ls = []

        layout = QFormLayout()

        # Conversion factor input
        self.lab_conv_fct = QLabel("Conversion Factor (um/pixel):")
        conv_fct_val = QDoubleValidator()
        conv_fct_val.setBottom(0)
        self.le_conv_fct = QLineEdit()
        self.le_conv_fct.setValidator(conv_fct_val)
        layout.addRow(self.lab_conv_fct, self.le_conv_fct)
        self.le_ls.append(self.le_conv_fct)

        # Bin width input
        self.lab_bin_width = QLabel("Bin Width (um):")
        bin_width_val = QIntValidator()
        bin_width_val.setBottom(0)
        bin_width_val.setTop(700)
        self.le_bin_width = QLineEdit()
        self.le_bin_width.setText("50")
        self.le_bin_width.setValidator(bin_width_val)
        layout.addRow(self.lab_bin_width, self.le_bin_width)
        self.le_ls.append(self.le_bin_width)

        # Upper limit input
        self.lab_up_lim = QLabel("Upper Limit (um):")
        up_lim_val = QIntValidator()
        up_lim_val.setBottom(0)
        self.le_up_lim = QLineEdit()
        self.le_up_lim.setText("700")
        self.le_up_lim.setValidator(up_lim_val)
        layout.addRow(self.lab_up_lim, self.le_up_lim)
        self.le_ls.append(self.le_up_lim)

        # Step size input
        self.lab_step_size = QLabel("Step Size (um):")
        step_size_val = QIntValidator()
        step_size_val.setBottom(0)
        self.le_step_size = QLineEdit()
        self.le_step_size.setText("5")
        self.le_step_size.setValidator(step_size_val)
        layout.addRow(self.lab_step_size, self.le_step_size)
        self.le_ls.append(self.le_step_size)

        # Channel names input
        self.lab_channel_names = QLabel("Channel Name(s):")
        self.le_channel_names = QLineEdit()
        self.le_channel_names.setPlaceholderText("e.g.  AF488, AF594")
        layout.addRow(self.lab_channel_names, self.le_channel_names)
        self.le_ls.append(self.le_channel_names)

        # Normalization constants input
        self.lab_norm_constants = QLabel("Normalization constant(s):")
        self.le_norm_constants = QLineEdit()
        self.le_norm_constants.setPlaceholderText("e.g.  1, 0")
        norm_constants_val = QRegExpValidator(
            QRegExp("^[-,0-9]+$")
        )  # Numbers and commas only
        self.le_norm_constants.setValidator(norm_constants_val)
        layout.addRow(self.lab_norm_constants, self.le_norm_constants)
        self.le_ls.append(self.le_norm_constants)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)
        self.setWindowTitle("User Inputs for Intensity Analysis")


class UI(QMainWindow):
    """Main window for the Image Wizard application."""

    def __init__(self):
        super().__init__()
        self.resize(800, 600)
        self.setWindowTitle("Image Wizard (Beta)")

        # Image data storage
        self.image_path_list = []
        self.image_data_list = []
        self.channel_list = []
        self.display_level_list = []

        # ROI storage
        self.masks_dict = {}  # key: exclusion name, value: ROI outline
        self.masks_label_dict = {}
        self.hole = None
        self.hole_label = None

        # Temporary ROI drawing storage (cleared after each mask is saved)
        self.mask_counter = 1
        self.temp_roi_path = []
        self.temp_roi = None
        self.temp_roi_start = None

        # Image view panel (deferred initialization)
        self.imv = None
        self._image_view_initialized = False

        self.windows = []

        # GUI components
        self.id_label = QLabel("No Images")

        self.channel_box = QComboBox()
        self.channel_box.currentIndexChanged.connect(self.changeChannel)

        self.create_hole_button = QPushButton("Set Implant Hole")
        self.create_hole_button.clicked.connect(self.drawHole)
        self.create_hole_button.setEnabled(False)

        self.create_mask_button = QPushButton("Define Exclusion")
        self.create_mask_button.clicked.connect(self.drawMask)
        self.create_mask_button.setEnabled(False)

        self.view_mask_button = QPushButton("View")
        self.view_mask_button.setEnabled(False)
        self.view_mask_button.clicked.connect(self.viewMask)
        self.view_mask_button.setShortcut("Ctrl+V")

        self.delete_mask_button = QPushButton("Delete")
        self.delete_mask_button.setStyleSheet("color: red")
        self.delete_mask_button.setEnabled(False)
        self.delete_mask_button.clicked.connect(self.deleteMask)

        self.mask_ls = QListWidget()
        self.mask_ls.addItem(
            QListWidgetItem("foo")
        )  # Dummy item to preserve widget size
        self.mask_ls.setMaximumWidth(self.mask_ls.sizeHintForColumn(0) * 5)
        self.mask_ls.clear()

        # Layout setup
        layout = QGridLayout()
        widget = QWidget()
        widget.setLayout(layout)
        layout.addWidget(self.id_label, 0, 0, Qt.AlignRight)
        layout.addWidget(QLabel("Select channel:"), 1, 0, Qt.AlignRight)
        layout.addWidget(self.channel_box, 1, 1, Qt.AlignLeft)
        layout.addWidget(self.create_hole_button, 0, 4, Qt.AlignLeft)
        layout.addWidget(self.create_mask_button, 1, 4, Qt.AlignLeft)
        layout.addWidget(self.mask_ls, 2, 4, 1, 1)

        # View/delete buttons sub-layout
        ls_button_widget = QWidget()
        ls_button_layout = QGridLayout(ls_button_widget)
        ls_button_layout.addWidget(self.view_mask_button, 0, 0)
        ls_button_layout.addWidget(self.delete_mask_button, 0, 1)
        layout.addWidget(ls_button_widget, 3, 4, Qt.AlignLeft)

        self.setCentralWidget(widget)

        # Menu actions
        self.open_action = QAction("&Open New Set", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.safelyOpenNewSet)

        self.close_action = QAction("&Close Application", self)
        self.close_action.setShortcut("Ctrl+W")
        self.close_action.triggered.connect(self.quitApp)

        self.save_action = QAction("Save Configuration", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.saveConfig)
        self.save_action.setEnabled(False)

        self.int_analysis = QAction("Analyze Stain Intensity", self)
        self.int_analysis.triggered.connect(self.askInput)
        self.int_analysis.setEnabled(True)

        self.cp_compile = QAction("Compile Cellpose Result")
        self.cp_compile.setEnabled(False)

        # Menu bar
        menuBar = self.menuBar()
        file_menu = menuBar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.close_action)

        edit_menu = menuBar.addMenu("&Edit")
        edit_menu.addAction(self.save_action)

        analysis_menu = menuBar.addMenu("&Analysis")
        analysis_menu.addAction(self.int_analysis)
        analysis_menu.addAction(self.cp_compile)

    def _initializeImageView(self):
        """Initialize the PyQtGraph ImageView after QApplication is ready."""
        if not self._image_view_initialized:
            self.imv = pg.ImageView()
            self.imv.getView().setMenuEnabled(False)
            self.imv.ui.roiBtn.hide()
            self.imv.ui.menuBtn.hide()
            self.imv.getHistogramWidget().sigLevelChangeFinished.connect(
                self.changeHistogramDisplay
            )

            layout = self.centralWidget().layout()
            layout.addWidget(self.imv, 2, 0, 3, 3)
            self._image_view_initialized = True

    def clearUp(self):
        """Reset application state when loading new images."""
        self.image_path_list = []
        self.image_data_list = []
        self.channel_list = []
        self.display_level_list = []

        self.save_action.setEnabled(False)
        self.id_label.setText("No Images")
        self.channel_box.clear()

        if self._image_view_initialized and self.imv is not None:
            self.imv.clear()

        self.mask_counter = 1

        if self._image_view_initialized and self.imv is not None:
            for _, roi in self.masks_dict.items():
                self.imv.removeItem(roi)

        self.masks_dict = {}

        if (
            self.hole is not None
            and self._image_view_initialized
            and self.imv is not None
        ):
            self.imv.removeItem(self.hole)
            self.imv.removeItem(self.hole_label)
            self.hole = None

        self.mask_ls.clear()

        if self._image_view_initialized and self.imv is not None:
            for _, label in self.masks_label_dict.items():
                self.imv.removeItem(label)

        self.masks_label_dict = {}
        self.hole_label = None

    def safelyOpenNewSet(self):
        """Safely open new image set with user confirmation if data exists."""
        if self.image_data_list:
            response = self.alert(
                "Unsaved changes will be lost.  Do you want to open a new set?"
            )
            if response == QMessageBox.Yes:
                self.openNewSet()
        else:
            self.openNewSet()

    def openNewSet(self):
        """Load a new set of TIFF images into the application."""
        fname = QFileDialog.getOpenFileNames(self, "", "", "TIFF Files (*.tif)")
        if fname[0]:
            self.clearUp()

            id_ls = []
            id = None

            for f in sorted(fname[0], key=str.lower):
                self.image_path_list.append(Path(f))
                id_ls.append(Path(f).parent.name)
                self.display_level_list.append(None)

            if len(set(id_ls)) == 1:
                id = list(set(id_ls))[0]
            else:
                print("Error: Image IDs do not match.")

            progress = QProgressDialog(
                "Loading Images...", "", 0, len(self.image_path_list)
            )
            progress.setCancelButton(None)
            progress.setWindowTitle("Progress")
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()

            # Extract channel names by finding common filename parts
            temp_split = []
            for p in self.image_path_list:
                temp_split.append(p.stem.split("_"))

            common_parts = set(temp_split[0]).intersection(*temp_split)

            for i, path in enumerate(self.image_path_list, start=1):
                ch_name = [x for x in path.stem.split("_") if x not in common_parts]
                self.channel_list.append("_".join(ch_name))
                self.image_data_list.append(np.array(tiff.imread(path)).T)
                progress.setValue(i)
                QApplication.processEvents()

            self.channel_box.addItems(self.channel_list)
            self.id_label.setText(id)

            if not self._image_view_initialized:
                self._initializeImageView()
            if self.imv is not None:
                self.imv.setImage(self.image_data_list[0])

            # Load existing configuration if available
            for config in self.image_path_list[0].parent.rglob("*.pickle"):
                self.loadConfig(config)

            self.buttonsEnabled(True)

    def buttonsEnabled(self, enabled):
        """Enable or disable all buttons to prevent errors during operations."""
        self.create_hole_button.setEnabled(enabled)
        self.create_mask_button.setEnabled(enabled)
        self.view_mask_button.setEnabled(enabled)
        self.delete_mask_button.setEnabled(enabled)
        self.save_action.setEnabled(enabled)

    def finishDrawing(self):
        """Clean up temporary ROI drawing state and re-enable buttons."""
        self.temp_roi = None
        self.temp_roi_path = []
        if self.imv is not None and self.temp_roi_start is not None:
            self.imv.removeItem(self.temp_roi_start)
        self.temp_roi_start = None

        if self.imv is not None:
            self.imv.scene.sigMouseClicked.disconnect()

        self.buttonsEnabled(True)

    def changeHistogramDisplay(self):
        """Store display levels when user adjusts histogram contrast."""
        if self.channel_box.count() != 0 and self.imv is not None:
            self.display_level_list[self.channel_box.currentIndex()] = (
                self.imv.getHistogramWidget().getLevels()
            )

    def changeChannel(self):
        """Update displayed image when user switches channels."""
        if self.channel_box.count() != 0 and self.imv is not None:
            self.imv.setImage(
                self.image_data_list[self.channel_box.currentIndex()],
                autoRange=False,
                levels=self.display_level_list[self.channel_box.currentIndex()],
            )

    def drawHole(self):
        """Start drawing an implant hole (red outline)."""
        color = "r"

        if self.hole is not None:
            response = self.alert("A hole already exists.  Do you want to replace it?")
            if response == QMessageBox.Yes:
                self.imv.removeItem(self.hole)
                self.imv.removeItem(self.hole_label)
                self.hole = None
                self.hole_label = None
                self.mask_ls.takeItem(0)

                self.imv.scene.sigMouseClicked.connect(
                    lambda event: self.polyLine(event, color)
                )
                self.buttonsEnabled(False)
        else:
            self.imv.scene.sigMouseClicked.connect(
                lambda event: self.polyLine(event, color)
            )
            self.buttonsEnabled(False)

    def drawMask(self):
        """Start drawing an exclusion mask (yellow outline)."""
        color = "y"
        self.imv.scene.sigMouseClicked.connect(
            lambda event: self.polyLine(event, color)
        )
        self.buttonsEnabled(False)

    def annotateMask(self, text, key):
        """Add a text label to a drawn mask at its center point."""
        label = pg.TextItem(
            text=text, anchor=(0.5, 0.5), fill=pg.mkBrush("w"), color="k"
        )
        label.setZValue(10)

        if key is None:  # Hole
            points = self.hole.getState()["points"]
            self.hole_label = label
        else:  # Exclusion mask
            points = self.masks_dict[key].getState()["points"]
            self.masks_label_dict[key] = label

        center = np.mean(points, axis=0)
        label.setPos(center[0], center[1])
        self.imv.addItem(label)

    def updateLabel(self, roi, key):
        """Update label position when user drags ROI handles."""
        new_pos = np.mean(roi.getState()["points"], axis=0)
        if key is None:
            self.hole_label.setPos(new_pos[0], new_pos[1])
        else:
            self.masks_label_dict[key].setPos(new_pos[0], new_pos[1])

    def saveMask(self):
        """Save the currently drawn exclusion mask."""
        mask_key = "Exclusion " + str(self.mask_counter)
        self.masks_dict[mask_key] = self.temp_roi

        self.annotateMask(str(self.mask_counter), mask_key)
        self.masks_dict[mask_key].sigRegionChangeFinished.connect(
            lambda roi: self.updateLabel(roi, mask_key)
        )

        self.mask_counter += 1
        self.mask_ls.addItem(mask_key)
        self.finishDrawing()

    def saveHole(self):
        """Save the currently drawn hole."""
        self.hole = self.temp_roi
        self.annotateMask("H", None)
        self.hole.sigRegionChangeFinished.connect(
            lambda roi: self.updateLabel(roi, None)
        )
        self.mask_ls.insertItem(0, "Hole")
        self.finishDrawing()

    def viewMask(self):
        """Zoom to the selected ROI in the image view."""
        for item in self.mask_ls.selectedItems():
            mask_key = item.text()

            if mask_key == "Hole":
                rect = self.hole.parentBounds()
            else:
                rect = self.masks_dict[mask_key].parentBounds()

            # Add padding around the ROI
            padding = min(rect.height(), rect.width()) * 2
            self.imv.getView().setRange(
                rect.adjusted(-padding, -padding, padding, padding)
            )

    def deleteMask(self):
        """Delete the selected mask or hole after user confirmation."""
        if not self.mask_ls.selectedItems():
            return

        response = self.alert(
            "The selected item will be deleted permanently.  Are you sure?"
        )
        if response == QMessageBox.Yes:
            for item in self.mask_ls.selectedItems():
                mask_key = item.text()

                if mask_key == "Hole":
                    self.imv.removeItem(self.hole)
                    self.imv.removeItem(self.hole_label)
                    self.mask_ls.takeItem(0)
                    self.hole = None
                    self.hole_label = None
                else:
                    self.imv.removeItem(self.masks_dict[mask_key])
                    self.imv.removeItem(self.masks_label_dict[mask_key])
                    self.masks_dict.pop(mask_key, None)
                    self.masks_label_dict.pop(mask_key, None)
                    self.mask_ls.takeItem(self.mask_ls.row(item))

    def polyLine(self, event, color):
        """Handle polygon drawing with mouse clicks. Left click adds points, right click finishes."""
        if event.button() == 1:  # Left click - add point
            scene_pos = event.scenePos()
            pos = self.imv.getView().mapSceneToView(scene_pos)
            self.temp_roi_path.append([pos.x(), pos.y()])

            if len(self.temp_roi_path) == 1:  # First point
                self.temp_roi = pg.PolyLineROI(
                    self.temp_roi_path,
                    movable=False,
                    pen=pg.mkPen(cosmetic=True, width=4, color=color),
                )
                # Mark starting point
                self.temp_roi_start = pg.ScatterPlotItem()
                self.temp_roi_start.addPoints(
                    pos=[pos], symbol="+", pen=pg.mkPen(color=color)
                )
                self.imv.addItem(self.temp_roi_start)

            elif len(self.temp_roi_path) == 2:  # Second point
                self.temp_roi.addFreeHandle(pos)
                self.temp_roi.addSegment(
                    self.temp_roi.getHandles()[-2], self.temp_roi.getHandles()[-1]
                )
                self.imv.addItem(self.temp_roi)

            elif len(self.temp_roi_path) > 2:  # Additional points
                self.temp_roi.addFreeHandle(pos)
                self.temp_roi.addSegment(
                    self.temp_roi.getHandles()[-2], self.temp_roi.getHandles()[-1]
                )

        elif event.button() == 2:  # Right click - finish polygon
            self.temp_roi.setPoints(self.temp_roi.getState()["points"], closed=True)
            if color == "r":
                self.saveHole()
            elif color == "y":
                self.saveMask()

    def saveConfig(self):
        """Save configuration file containing holes and exclusion masks."""
        num_hole = 1 if self.hole is not None else 0
        num_masks = len(self.masks_dict)

        if num_hole == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Implant hole is not defined. ")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        confirm_msg = (
            f"Number of Hole: {num_hole}\n"
            f"Number of Exclusions: {num_masks}\n\n"
            "Are you sure you want to save?"
        )
        response = self.alert(confirm_msg)

        if response == QMessageBox.Yes:
            has_config = any(self.image_path_list[0].parent.rglob("*.pickle"))

            if has_config:
                overwrite = self.alert(
                    "Existing configuration found.  Do you want to overwrite it?"
                )
                if overwrite == QMessageBox.Yes:
                    self.saveH5Mask(num_hole, num_masks)
                    self.saveOutline()
            else:
                self.saveH5Mask(num_hole, num_masks)
                self.saveOutline()

    def saveOutline(self):
        """Save ROI outlines to pickle file for later recovery."""
        image_id = self.image_path_list[0].parent.name
        pkl_fname = f"{image_id}_config.pickle"
        pkl_path = self.image_path_list[0].parent / pkl_fname

        master_dict = {
            "hole": self.hole.saveState() if self.hole is not None else None,
            "exclusions_states": [
                item.saveState() for item in self.masks_dict.values()
            ],
        }

        with open(pkl_path, "wb") as f:
            pkl.dump(master_dict, f)

    def saveH5Mask(self, num_hole, num_masks):
        """Save masks as binary maps in HDF5 format."""
        progress = QProgressDialog("Saving Changes...", "", 0, num_hole + num_masks)
        progress.setCancelButton(None)
        progress.setWindowTitle("Progress")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        cmap_hole = colors.ListedColormap(["#FF000000", "r"])
        cmap_mask = colors.ListedColormap(["#FF000000", "y"])

        image_id = self.image_path_list[0].parent.name
        h5_fname = f"{image_id}_mask.h5"
        img_fname = f"{image_id}_preview.png"
        h5_path = self.image_path_list[0].parent / h5_fname
        img_path = self.image_path_list[0].parent / img_fname

        hf = h5py.File(h5_path, "w")

        nx, ny = self.image_data_list[0].shape[:2]
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T  # Coordinates for path containment check

        if self.hole is not None:
            coord_hole = self.hole.saveState()["points"]
            path = PltPath(coord_hole)
            grid = path.contains_points(points).reshape((ny, nx))
            hf.create_dataset("hole", data=grid, compression="gzip")
        else:
            hf.create_dataset("hole", data=np.full((ny, nx), False), compression="gzip")

        progress.setValue(1)

        if self.masks_dict:
            grid = np.full((ny, nx), False)
            for mask in self.masks_dict.values():
                path = PltPath(mask.saveState()["points"])
                temp_grid = path.contains_points(points).reshape((ny, nx))
                grid = np.logical_or(grid, temp_grid)

                progress.setValue(progress.value() + 1)
                QApplication.processEvents()

            hf.create_dataset("exclusions", data=grid, compression="gzip")
        else:
            hf.create_dataset(
                "exclusions", data=np.full((ny, nx), False), compression="gzip"
            )

        # Create preview plot
        plt.imshow(hf["hole"][:], cmap=cmap_hole)
        plt.imshow(hf["exclusions"][:], cmap=cmap_mask)
        plt.savefig(img_path, dpi=200)
        plt.close()

        hf.close()

    def loadConfig(self, path):
        """Load previous configuration from pickle file."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Masks are loaded from an existing configuration.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

        with open(path, "rb") as config:
            master_dict = pkl.load(config)

        h_state = master_dict["hole"]
        if h_state is not None:
            self.hole = pg.PolyLineROI(
                h_state["points"],
                movable=False,
                pen=pg.mkPen(cosmetic=True, width=4, color="r"),
                closed=True,
            )
            self.imv.addItem(self.hole)
            self.annotateMask("H", None)
            self.mask_ls.insertItem(0, "Hole")

        mask_states = master_dict["exclusions_states"]
        if mask_states:
            for state in mask_states:
                key = f"Exclusion {self.mask_counter}"
                self.masks_dict[key] = pg.PolyLineROI(
                    state["points"],
                    movable=False,
                    pen=pg.mkPen(cosmetic=True, width=4, color="y"),
                    closed=True,
                )
                self.imv.addItem(self.masks_dict[key])
                self.annotateMask(str(self.mask_counter), key)
                self.mask_ls.addItem(key)
                self.mask_counter += 1

    def askInput(self):
        """Show dialog to collect user inputs for intensity analysis."""
        self.intensity_dialog = InputDialogIntensity(self)
        self.windows.append(self.intensity_dialog)
        self.intensity_dialog.show()
        self.intensity_dialog.buttonBox.accepted.connect(self.finishInput)

    def finishInput(self):
        """Validate user inputs and start intensity analysis."""
        self.int_analysis.setEnabled(False)

        input_ok = True
        dict_keys = ["conv_fct", "bin", "up_lim", "step", "chl_names", "norm"]

        # Check if all fields are completed
        for line_edit, key in zip(self.intensity_dialog.le_ls, dict_keys):
            if line_edit.text():
                self.intensity_dialog.dict[key] = line_edit.text()
            else:
                input_ok = False

        if input_ok:
            # Parse and clean channel names and normalization constants
            channels = [
                ch
                for ch in self.intensity_dialog.dict["chl_names"]
                .replace(" ", "")
                .split(",")
                if ch
            ]
            norms = [
                norm
                for norm in self.intensity_dialog.dict["norm"]
                .replace(" ", "")
                .split(",")
                if norm
            ]

            if len(channels) == len(norms):
                self.intensity_dialog.dict["chl_names"] = channels
                self.intensity_dialog.dict["norm"] = norms
                self.intensity_dialog.close()
                self.intensityAnalysis(self.intensity_dialog.dict)
            else:
                self._showError("Channel names and normalization constants must match.")
        else:
            self._showError("All fields are required.")

    def _showError(self, message):
        """Show error message dialog."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def intensityAnalysis(self, params):
        """Perform intensity analysis on images using user-specified parameters."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open the folder containing all images"
        )

        if not folder_path:
            self.int_analysis.setEnabled(True)
            return

        data_path = Path(folder_path)

        def match_id(identifier, file_ext, search_path):
            """Find file matching identifier and extension. Returns last match if multiple found."""
            matches = list(Path(search_path).rglob(f"*{identifier}*{file_ext}"))
            return matches[-1] if matches else None

        def unpack_h5(file_path):
            """Extract hole and combined mask data from HDF5 file."""
            with h5py.File(file_path, "r") as f:
                return f["hole"][:], np.logical_or(f["hole"][:], f["exclusions"][:])

        # Initialize data lists
        image_ids = []
        mask_paths = []
        tiff_paths = []
        valid_ids = []

        # Parse user parameters
        upper_limit_um = int(params["up_lim"])
        step_size = int(params["step"])
        channels = params["chl_names"]
        norm_methods = [eval(i) for i in params["norm"]]
        conv_factor = float(params["conv_fct"])
        bin_width_um = int(params["bin"])

        bins = np.arange(0, upper_limit_um + step_size, step_size)

        # Find all mask files and corresponding images
        for mask_path in data_path.rglob("*_mask.h5"):
            mask_paths.append(mask_path)
            image_id = mask_path.parent.name
            image_ids.append(image_id)

            # Find corresponding TIFF files for each channel
            channel_paths = []
            for channel in channels:
                channel_paths.append(match_id(channel, ".tif", mask_path.parent))

            # Only include if all channels are found
            if None in channel_paths:
                tiff_paths.append(None)
            else:
                tiff_paths.append(channel_paths)

        # Create batch array and filter out incomplete data
        batch = np.array([image_ids, mask_paths, tiff_paths], dtype=object).T
        batch = batch[~np.any(batch is None, axis=1)]

        if len(batch) == 0:
            self._showError("No images/masks were found. Please start over.")
            self.int_analysis.setEnabled(True)
            return

        # Show analysis summary
        summary_lines = [
            f"Masks found for: {len(batch)} images.",
            "Normalization constants:",
        ]
        summary_lines.extend(
            [f"{ch}: {norm}" for ch, norm in zip(channels, norm_methods)]
        )
        summary_text = "\n".join(summary_lines)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(summary_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

        # Initialize results storage
        results_master = [[] for _ in range(len(channels))]
        valid_ids = []

        progress = QProgressDialog("Analyzing intensity...", "", 0, len(batch))
        progress.setCancelButton(None)
        progress.setWindowTitle("Progress")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        counter = 1

        # Process each image in the batch
        for image_id, mask_file, channel_files in batch:
            map_hole = None
            mask_all = None

            try:
                map_hole, mask_all = unpack_h5(mask_file)
            except Exception:
                print(image_id, "mask file broken")
                continue

            valid_ids.append(image_id)

            # Calculate distance from hole for each pixel
            dist_2d_pixels = ndimage.distance_transform_edt(map_hole - 1)
            dist_1d_um = (
                np.ma.array(dist_2d_pixels, mask=mask_all).compressed()
            ) * conv_factor

            intensity_results = []

            # Measure intensity for each channel
            for path in channel_files:
                img = tiff.imread(path)
                intensity_1d = np.ma.array(img, mask=mask_all).compressed()

                bin_means, _, _ = stats.binned_statistic(
                    dist_1d_um, intensity_1d, statistic="mean", bins=bins
                )
                intensity_results.append(bin_means)

            # Create and save intensity plot
            fig, axes = plt.subplots(nrows=len(channels), figsize=(10, 6))
            fig.suptitle(f"{image_id} intensity plot")
            axes_arr = np.array(axes)

            for i, ax in enumerate(axes_arr.flatten()):
                ax.plot(bins[1:], intensity_results[i], label=channels[i])
                ax.legend(fontsize="large")
                ax.set_ylabel("Intensity")
                ax.set_xlabel("Distance (micron)")

            plt.tight_layout()
            plt.savefig(mask_file.parent / f"{image_id}_intensity-plot.png", dpi=200)
            plt.close()

            # Compile results for each channel
            for i, channel_data in enumerate(intensity_results):
                reshaped = np.sum(
                    np.reshape(
                        channel_data,
                        (upper_limit_um // bin_width_um, bin_width_um // step_size),
                    ),
                    axis=1,
                )
                # Normalize intensities
                normalized = reshaped / reshaped[-1] - (1 - norm_methods[i])
                results_master[i].append(np.clip(normalized, 0, None))

            progress.setValue(counter)
            QApplication.processEvents()
            counter += 1

        # Export results to Excel
        dist_range = np.arange(0, upper_limit_um, bin_width_um)
        headers = [f"{x}-{x + bin_width_um}" for x in dist_range]

        output_filename = f"{time.strftime('%Y%m%d-%H%M%S')}_intensity-raw-output.xlsx"
        with pd.ExcelWriter(data_path / output_filename) as writer:
            for i, channel in enumerate(channels):
                df = pd.DataFrame(
                    data=results_master[i], index=valid_ids, columns=headers
                )
                df.index.name = "id"
                df.to_excel(writer, sheet_name=channel)

        self.int_analysis.setEnabled(True)

        # Show completion message
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Intensity analysis finished.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def quitApp(self):
        """Close the application after user confirmation."""
        response = self.alert("Are you sure you want to exit the application?")
        if response == QMessageBox.Yes:
            sys.exit()

    def closeEvent(self, event):
        """Handle window close event with user confirmation."""
        response = self.alert("Are you sure you want to exit the application?")
        if response == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def alert(self, text):
        """Show warning dialog with Yes/Cancel options."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning")
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        return msg.exec_()


def main():
    """Main function to run the Image Wizard application."""
    global pg

    app = QApplication(sys.argv)

    # Import pyqtgraph after QApplication creation to avoid widget creation errors
    import pyqtgraph as pyqt_graph

    pg = pyqt_graph

    window = UI()
    window._initializeImageView()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
