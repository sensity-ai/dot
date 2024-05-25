#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import os
import sys
import tkinter
import traceback
from pathlib import Path

import click
import customtkinter
import yaml

from dot.__main__ import run

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tkinter.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tkinter.Label(
            tw,
            text=self.text,
            justify=tkinter.LEFT,
            background="#ffffff",
            relief=tkinter.SOLID,
            borderwidth=1,
            font=("arial", "10", "normal"),
        )
        label.pack(ipadx=8, ipady=5)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class ToplevelUsageWindow(customtkinter.CTkToplevel):
    """
    The class of the usage window
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Usage")
        self.geometry(f"{700}x{470}")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.textbox = customtkinter.CTkTextbox(
            master=self, width=700, height=550, corner_radius=0
        )
        self.textbox.grid(row=0, column=0, sticky="nsew")
        self.textbox.insert(
            "0.0",
            """
            source (str): The source image or video.\n
            target (Union[int, str]): The target image, video or camera id.\n
            config_file (str): Path to the configuration file for the deepfake.\n
            swap_type (str): The type of swap to run.\n
            gpen_type (str, optional): The type of gpen model to use. Defaults to None.\n
            gpen_path (str, optional): The path to the gpen models. Defaults to "saved_models/gpen".\n
            show_fps (bool, optional): Pass flag to show fps value. Defaults to False.\n
            use_gpu (bool, optional): Pass flag to use GPU else use CPU. Defaults to False.\n
            head_pose (bool): Estimates head pose before swap. Used by fomm.\n
            model_path (str, optional): The path to the model's weights. Defaults to None.\n
            parsing_model_path (str, optional): The path to the parsing model. Defaults to None.\n
            arcface_model_path (str, optional): The path to the arcface model. Defaults to None.\n
            checkpoints_dir (str, optional): The path to the checkpoints directory. Defaults to None.\n
            crop_size (int, optional): The size to crop the images to. Defaults to 224.\n
            """,
        )
        self.textbox.configure(state=tkinter.DISABLED)


class ToplevelAboutWindow(customtkinter.CTkToplevel):
    """
    The class of the about window
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("About DOT")
        self.geometry(f"{700}x{300}")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.textbox = customtkinter.CTkTextbox(
            master=self, width=700, height=300, corner_radius=0
        )
        self.textbox.grid(row=0, column=0, sticky="nsew")

        self.textbox.insert(
            "0.0",
            """
            DOT (aka Deepfake Offensive Toolkit) makes real-time, controllable deepfakes ready for virtual \n
            cameras injection. DOT is created for performing penetration testing against e.g. identity \n
            verification and video conferencing systems, for the use by security analysts, \n
            Red Team members, and biometrics researchers. DOT is developed for research and demonstration purposes. \n
            As an end user, you have the responsibility to obey all applicable laws when using this program. \n
            Authors and contributing developers assume no liability and are not responsible for any misuse \n
            or damage caused by the use of this program.
            """,
        )
        self.textbox.configure(state=tkinter.DISABLED)


class App(customtkinter.CTk):
    """
    The main class of the ui interface
    """

    def __init__(self):
        super().__init__()

        # configure window
        self.title("Deepfake Offensive Toolkit")
        self.geometry(f"{835}x{600}")
        self.resizable(False, False)

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # create menubar
        menubar = tkinter.Menu(self)

        filemenu = tkinter.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tkinter.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Usage", command=self.usage_window)
        helpmenu.add_separator()
        helpmenu.add_command(label="About DOT", command=self.about_window)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.config(menu=menubar)

        self.toplevel_usage_window = None
        self.toplevel_about_window = None

        # create entry text for source, target and config
        self.source_target_config_frame = customtkinter.CTkFrame(self)
        self.source_target_config_frame.grid(
            row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew"
        )

        self.source_label = customtkinter.CTkLabel(
            master=self.source_target_config_frame, text="source"
        )

        self.source = customtkinter.CTkEntry(
            master=self.source_target_config_frame,
            placeholder_text="source",
            width=85,
        )
        self.source_button = customtkinter.CTkButton(
            master=self.source_target_config_frame,
            fg_color="gray",
            text_color="white",
            text="Open",
            command=lambda: self.UploadAction(self.source),
            width=10,
        )

        self.target = customtkinter.CTkEntry(
            master=self.source_target_config_frame, placeholder_text="target", width=85
        )
        self.target_label = customtkinter.CTkLabel(
            master=self.source_target_config_frame, text="target"
        )

        self.config_file_var = customtkinter.StringVar(
            value="Select"
        )  # set initial value

        self.config_file_combobox = customtkinter.CTkOptionMenu(
            master=self.source_target_config_frame,
            values=["fomm", "faceswap_cv2", "simswap", "simswaphq"],
            command=self.optionmenu_callback,
            variable=self.config_file_var,
            width=85,
            button_color="#3C3C3C",
            fg_color="#343638",
            dynamic_resizing=False,
        )

        self.config_file = customtkinter.CTkEntry(
            master=self.source_target_config_frame, placeholder_text="config", width=85
        )
        self.config_file_label = customtkinter.CTkLabel(
            master=self.source_target_config_frame, text="config_file"
        )
        self.config_file_button = customtkinter.CTkButton(
            master=self.source_target_config_frame,
            fg_color="gray",
            text_color="white",
            text="Open",
            command=lambda: self.upload_action_config_file(
                self.config_file_combobox, self.config_file_var
            ),
            width=10,
        )

        self.source_label.grid(row=1, column=0, pady=(32, 10), padx=30, sticky="w")
        self.source.grid(row=1, column=0, pady=(32, 10), padx=(80, 20), sticky="w")
        self.source_button.grid(
            row=1,
            column=0,
            pady=(32, 10),
            padx=(175, 20),
            sticky="w",
        )
        self.CreateToolTip(
            self.source,
            text="The path of the source directory that contains a set of images\n"
            "or the path of one image intended for utilization in the deepfake generation process",
        )

        self.target.grid(row=2, column=0, pady=10, padx=(80, 20), sticky="w")
        self.target_label.grid(row=2, column=0, pady=10, padx=(35, 20), sticky="w")
        self.target.insert(0, 0)
        self.CreateToolTip(
            self.target, text="The camera id. Usually 0 is the correct id"
        )

        self.config_file_combobox.grid(
            row=3, column=0, pady=10, padx=(80, 20), sticky="w"
        )
        self.config_file_label.grid(row=3, column=0, pady=10, padx=10, sticky="w")

        self.config_file_button.grid(
            row=3,
            column=0,
            pady=10,
            padx=(175, 20),
            sticky="w",
        )
        self.CreateToolTip(
            self.config_file_combobox, text="Configuration file for the deepfake"
        )

        # create entry text for dot options
        self.option_entry_frame = customtkinter.CTkFrame(self)
        self.option_entry_frame.grid(
            row=1, column=0, columnspan=4, padx=(20, 20), pady=(20, 0), sticky="nsew"
        )

        self.advanced_options = customtkinter.CTkLabel(
            master=self.option_entry_frame, text="Advanced"
        )

        self.model_path_label = customtkinter.CTkLabel(
            master=self.option_entry_frame, text="model_path"
        )
        self.model_path = customtkinter.CTkEntry(
            master=self.option_entry_frame, placeholder_text="model_path", width=85
        )

        self.parsing_model_path_label = customtkinter.CTkLabel(
            master=self.option_entry_frame, text="parsing_model"
        )
        self.parsing_model_path = customtkinter.CTkEntry(
            master=self.option_entry_frame,
            placeholder_text="parsing_model_path",
            width=85,
        )

        self.arcface_model_path_label = customtkinter.CTkLabel(
            master=self.option_entry_frame, text="arcface_model"
        )
        self.arcface_model_path = customtkinter.CTkEntry(
            master=self.option_entry_frame,
            placeholder_text="arcface_model_path",
            width=85,
        )

        self.checkpoints_dir_label = customtkinter.CTkLabel(
            master=self.option_entry_frame, text="checkpoints_dir"
        )
        self.checkpoints_dir = customtkinter.CTkEntry(
            master=self.option_entry_frame, placeholder_text="checkpoints_dir", width=85
        )

        self.gpen_path_label = customtkinter.CTkLabel(
            master=self.option_entry_frame, text="gpen_path"
        )
        self.gpen_path = customtkinter.CTkEntry(
            master=self.option_entry_frame, placeholder_text="gpen_path", width=85
        )

        self.crop_size_label = customtkinter.CTkLabel(
            master=self.option_entry_frame, text="crop_size"
        )
        self.crop_size = customtkinter.CTkEntry(
            master=self.option_entry_frame, placeholder_text="crop_size"
        )

        self.model_path_button = customtkinter.CTkButton(
            master=self.option_entry_frame,
            fg_color="gray",
            text_color="white",
            text="Open",
            command=lambda: self.UploadAction(self.model_path),
            width=10,
        )
        self.parsing_model_path_button = customtkinter.CTkButton(
            master=self.option_entry_frame,
            fg_color="gray",
            text_color="white",
            text="Open",
            command=lambda: self.UploadAction(self.parsing_model_path),
            width=10,
        )
        self.arcface_model_path_button = customtkinter.CTkButton(
            master=self.option_entry_frame,
            fg_color="gray",
            text_color="white",
            text="Open",
            command=lambda: self.UploadAction(self.arcface_model_path),
            width=10,
        )
        self.checkpoints_dir_button = customtkinter.CTkButton(
            master=self.option_entry_frame,
            fg_color="gray",
            text_color="white",
            text="Open",
            command=lambda: self.UploadAction(self.checkpoints_dir),
            width=10,
        )
        self.gpen_path_button = customtkinter.CTkButton(
            master=self.option_entry_frame,
            fg_color="gray",
            text_color="white",
            text="Open",
            command=lambda: self.UploadAction(self.gpen_path),
            width=10,
        )

        self.advanced_options.grid(row=0, column=0, pady=10, padx=(20, 20), sticky="w")

        self.model_path_label.grid(row=1, column=2, pady=10, padx=(40, 20), sticky="w")
        self.model_path.grid(row=1, column=2, pady=10, padx=(115, 20), sticky="w")
        self.model_path_button.grid(
            row=1,
            column=2,
            pady=10,
            padx=(210, 20),
            sticky="w",
        )

        self.parsing_model_path_label.grid(
            row=2, column=2, pady=10, padx=(23, 20), sticky="w"
        )
        self.parsing_model_path.grid(
            row=2, column=2, pady=10, padx=(115, 20), sticky="w"
        )
        self.parsing_model_path_button.grid(
            row=2,
            column=2,
            pady=10,
            padx=(210, 20),
            sticky="w",
        )

        self.arcface_model_path_label.grid(
            row=3, column=2, pady=10, padx=(21, 20), sticky="w"
        )
        self.arcface_model_path.grid(
            row=3, column=2, pady=10, padx=(115, 20), sticky="w"
        )
        self.arcface_model_path_button.grid(
            row=3,
            column=2,
            pady=10,
            padx=(210, 20),
            sticky="w",
        )

        self.checkpoints_dir_label.grid(
            row=1, column=3, pady=10, padx=(16, 20), sticky="w"
        )
        self.checkpoints_dir.grid(row=1, column=3, pady=10, padx=(115, 20), sticky="w")
        self.checkpoints_dir_button.grid(
            row=1,
            column=3,
            pady=10,
            padx=(210, 20),
            sticky="w",
        )

        self.gpen_path_label.grid(row=2, column=3, pady=10, padx=(48, 20), sticky="w")
        self.gpen_path.grid(row=2, column=3, pady=10, padx=(115, 20), sticky="w")
        self.gpen_path_button.grid(
            row=2,
            column=3,
            pady=10,
            padx=(210, 20),
            sticky="w",
        )

        self.crop_size_label.grid(row=3, column=3, pady=10, padx=(50, 20), sticky="w")
        self.crop_size.grid(row=3, column=3, pady=10, padx=(115, 20), sticky="w")

        self.CreateToolTip(self.crop_size, text="The size of the image crop")
        self.CreateToolTip(self.gpen_path, text="The path to the gpen models")
        self.CreateToolTip(
            self.checkpoints_dir,
            text="The path to the checkpoints directory. Used by SimSwap",
        )
        self.CreateToolTip(
            self.arcface_model_path,
            text="The path to the arcface model. Used by SimSwap",
        )
        self.CreateToolTip(
            self.parsing_model_path,
            text="The path to the parsing model. Used by SimSwap",
        )
        self.CreateToolTip(
            self.model_path, text="The path to the model's weights. Used by fomm"
        )

        # create radiobutton frame for swap_type
        self.swap_type_frame = customtkinter.CTkFrame(self)
        self.swap_type_frame.grid(
            row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew"
        )
        self.swap_type_radio_var = tkinter.StringVar(value=None)
        self.swap_type_label_radio_group = customtkinter.CTkLabel(
            master=self.swap_type_frame, text="swap_type"
        )
        self.swap_type_label_radio_group.grid(
            row=0, column=2, columnspan=1, padx=10, pady=10, sticky=""
        )
        self.fomm_radio_button = customtkinter.CTkRadioButton(
            master=self.swap_type_frame,
            variable=self.swap_type_radio_var,
            value="fomm",
            text="fomm",
        )
        self.fomm_radio_button.grid(row=1, column=2, pady=10, padx=20, sticky="w")
        self.CreateToolTip(self.fomm_radio_button, text="Use the deepfake from fomm")

        self.faceswap_cv2_radio_button = customtkinter.CTkRadioButton(
            master=self.swap_type_frame,
            variable=self.swap_type_radio_var,
            value="faceswap_cv2",
            text="faceswap_cv2",
        )
        self.faceswap_cv2_radio_button.grid(
            row=2, column=2, pady=10, padx=20, sticky="w"
        )
        self.CreateToolTip(
            self.faceswap_cv2_radio_button, text="Use the deepfake from faceswap cv2"
        )

        self.simswap_radio_button = customtkinter.CTkRadioButton(
            master=self.swap_type_frame,
            variable=self.swap_type_radio_var,
            value="simswap",
            text="simswap",
        )
        self.simswap_radio_button.grid(row=3, column=2, pady=10, padx=20, sticky="w")
        self.CreateToolTip(
            self.simswap_radio_button, text="Use the deepfake from SimSwap"
        )

        # create radiobutton frame for gpen_type
        self.gpen_type_frame = customtkinter.CTkFrame(self)
        self.gpen_type_frame.grid(
            row=0, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew"
        )
        self.gpen_type_radio_var = tkinter.StringVar(value="")
        self.gpen_type_label_radio_group = customtkinter.CTkLabel(
            master=self.gpen_type_frame, text="gpen_type"
        )
        self.gpen_type_label_radio_group.grid(
            row=0, column=2, columnspan=1, padx=10, pady=10, sticky=""
        )
        self.gpen_type_radio_button_1 = customtkinter.CTkRadioButton(
            master=self.gpen_type_frame,
            variable=self.gpen_type_radio_var,
            value="gpen_256",
            text="gpen_256",
        )

        self.gpen_type_radio_button_1.grid(
            row=1, column=2, pady=10, padx=20, sticky="w"
        )
        self.CreateToolTip(
            self.gpen_type_radio_button_1, text="Apply face restoration with GPEN 256"
        )

        self.gpen_type_radio_button_2 = customtkinter.CTkRadioButton(
            master=self.gpen_type_frame,
            variable=self.gpen_type_radio_var,
            value="gpen_512",
            text="gpen_512",
        )
        self.gpen_type_radio_button_2.grid(
            row=2, column=2, pady=10, padx=20, sticky="w"
        )
        self.CreateToolTip(
            self.gpen_type_radio_button_2, text="Apply face restoration with GPEN 512"
        )

        # create checkbox and switch frame
        self.checkbox_slider_frame = customtkinter.CTkFrame(self)
        self.checkbox_slider_frame.grid(
            row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew"
        )

        self.show_fps_checkbox_var = tkinter.IntVar()
        self.show_fps_checkbox = customtkinter.CTkCheckBox(
            master=self.checkbox_slider_frame,
            text="show_fps",
            variable=self.show_fps_checkbox_var,
        )

        self.use_gpu_checkbox_var = tkinter.IntVar()
        self.use_gpu_checkbox = customtkinter.CTkCheckBox(
            master=self.checkbox_slider_frame,
            text="use_gpu",
            variable=self.use_gpu_checkbox_var,
        )

        self.head_pose_checkbox_var = tkinter.IntVar()
        self.head_pose_checkbox = customtkinter.CTkCheckBox(
            master=self.checkbox_slider_frame,
            text="head_pose",
            variable=self.head_pose_checkbox_var,
        )

        self.show_fps_checkbox.grid(row=1, column=3, pady=(39, 0), padx=20, sticky="w")
        self.use_gpu_checkbox.grid(row=2, column=3, pady=(20, 0), padx=20, sticky="w")
        self.head_pose_checkbox.grid(row=5, column=3, pady=(20, 0), padx=20, sticky="w")
        self.CreateToolTip(self.show_fps_checkbox, text="Show the fps value")
        self.CreateToolTip(
            self.use_gpu_checkbox,
            text="If checked, the deepfake will use the GPU.\n"
            "If it's not checked, the deepfake will use the CPU",
        )
        self.CreateToolTip(
            self.head_pose_checkbox, text="Estimate head pose before swap. Used by fomm"
        )

        # create run button
        self.error_label = customtkinter.CTkLabel(
            master=self, text_color="red", text=""
        )
        self.error_label.grid(
            row=4, column=0, columnspan=4, padx=(20, 20), pady=(0, 20), sticky="nsew"
        )

        self.run_button = customtkinter.CTkButton(
            master=self,
            fg_color="white",
            border_width=2,
            text_color="black",
            text="RUN",
            command=lambda: self.start_button_event(self.error_label),
        )
        self.run_button.grid(
            row=2, column=1, columnspan=2, padx=(50, 150), pady=(20, 0), sticky="nsew"
        )
        self.CreateToolTip(self.run_button, text="Start running the deepfake")

        self.run_label = customtkinter.CTkLabel(
            master=self,
            text="The initial execution of dot may require a few minutes to complete.",
            text_color="gray",
        )
        self.run_label.grid(
            row=3, column=0, columnspan=3, padx=(180, 0), pady=(0, 20), sticky="nsew"
        )

        self.resources_path = ""

        # MacOS bundle has different resource directory structure
        if sys.platform == "darwin":
            if getattr(sys, "frozen", False):
                self.resources_path = os.path.join(
                    str(Path(sys.executable).resolve().parents[0]).replace("MacOS", ""),
                    "Resources",
                )

    def CreateToolTip(self, widget, text):
        toolTip = ToolTip(widget)

        def enter(event):
            toolTip.showtip(text)

        def leave(event):
            toolTip.hidetip()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def usage_window(self):
        """
        Open the usage window
        """

        if (
            self.toplevel_usage_window is None
            or not self.toplevel_usage_window.winfo_exists()
        ):
            self.toplevel_usage_window = ToplevelUsageWindow(
                self
            )  # create window if its None or destroyed
        self.toplevel_usage_window.focus()

    def about_window(self):
        """
        Open the about window
        """

        if (
            self.toplevel_about_window is None
            or not self.toplevel_about_window.winfo_exists()
        ):
            self.toplevel_about_window = ToplevelAboutWindow(
                self
            )  # create window if its None or destroyed
        self.toplevel_about_window.focus()

    def UploadAction(self, entry_element: customtkinter.CTkOptionMenu):
        """
        Action for the upload buttons to update the value of a CTkEntry

        Args:
            entry_element (customtkinter.CTkOptionMenu): The CTkEntry element.
        """

        filename = tkinter.filedialog.askopenfilename()
        self.modify_entry(entry_element, filename)

    def modify_entry(self, entry_element: customtkinter.CTkEntry, text: str):
        """
        Modify the value of the CTkEntry

        Args:
            entry_element (customtkinter.CTkOptionMenu): The CTkEntry element.
            text (str): The new text that will be inserted into the CTkEntry
        """

        entry_element.delete(0, tkinter.END)
        entry_element.insert(0, text)

    def upload_action_config_file(
        self,
        element: customtkinter.CTkOptionMenu,
        config_file_var: customtkinter.StringVar,
    ):
        """
        Set the configurations for the swap_type using the upload button

        Args:
            element (customtkinter.CTkOptionMenu): The OptionMenu element.
            config_file_var (customtkinter.StringVar): OptionMenu variable.
        """

        entry_list = [
            "source",
            "target",
            "model_path",
            "parsing_model_path",
            "arcface_model_path",
            "checkpoints_dir",
            "gpen_path",
            "crop_size",
        ]
        radio_list = ["swap_type"]

        filename = tkinter.filedialog.askopenfilename()

        config = {}
        if len(filename) > 0:
            with open(filename) as f:
                config = yaml.safe_load(f)

        if config["swap_type"] == "simswap":
            if config.get("swap_type", "0") == "512":
                config_file_var = "simswaphq"
            else:
                config_file_var = "simswap"
        else:
            config_file_var = config["swap_type"]

        element.set(config_file_var)

        for key in config.keys():
            if key in entry_list:
                self.modify_entry(eval(f"self.{key}"), config[key])
            elif key in radio_list:
                self.swap_type_radio_var = tkinter.StringVar(value=config[key])
                eval(f"self.{config[key]}_radio_button").invoke()

        for entry in entry_list:
            if entry not in ["source", "target"]:
                if entry not in config.keys():
                    self.modify_entry(eval(f"self.{entry}"), "")

    def optionmenu_callback(self, choice: str):
        """
        Set the configurations for the swap_type using the optionmenu

        Args:
            choice (str): The type of swap to run.
        """

        entry_list = ["source", "target", "crop_size"]
        radio_list = ["swap_type", "gpen_type"]
        model_list = [
            "model_path",
            "parsing_model_path",
            "arcface_model_path",
            "checkpoints_dir",
            "gpen_path",
        ]

        config_file = os.path.join(self.resources_path, f"configs/{choice}.yaml")

        if os.path.isfile(config_file):
            config = {}
            with open(config_file) as f:
                config = yaml.safe_load(f)

            for key in config.keys():
                if key in entry_list:
                    self.modify_entry(eval(f"self.{key}"), config[key])
                elif key in radio_list:
                    if key == "swap_type":
                        self.swap_type_radio_var = tkinter.StringVar(value=config[key])
                    elif key == "gpen_type":
                        self.gpen_type_radio_var = tkinter.StringVar(value=config[key])
                    eval(f"self.{config[key]}_radio_button").invoke()
                elif key in model_list:
                    self.modify_entry(
                        eval(f"self.{key}"),
                        os.path.join(self.resources_path, config[key]),
                    )

            for entry in entry_list:
                if entry not in ["source", "target"]:
                    if entry not in config.keys():
                        self.modify_entry(eval(f"self.{entry}"), "")

    def start_button_event(self, error_label):
        """
        Start running the deepfake
        """
        try:
            error_label.configure(text="")

            # load config, if provided
            config = {}
            if len(self.config_file.get()) > 0:
                with open(self.config_file.get()) as f:
                    config = yaml.safe_load(f)

            # run dot
            run(
                swap_type=config.get(
                    "swap_type", self.swap_type_radio_var.get() or None
                ),
                source=config.get("source", self.source.get() or None),
                target=config.get("target", self.target.get() or None),
                model_path=config.get("model_path", self.model_path.get() or None),
                parsing_model_path=config.get(
                    "parsing_model_path", self.parsing_model_path.get() or None
                ),
                arcface_model_path=config.get(
                    "arcface_model_path", self.arcface_model_path.get() or None
                ),
                checkpoints_dir=config.get(
                    "checkpoints_dir", self.checkpoints_dir.get() or None
                ),
                gpen_type=config.get("gpen_type", self.gpen_type_radio_var.get()),
                gpen_path=config.get(
                    "gpen_path", self.gpen_path.get() or "saved_models/gpen"
                ),
                crop_size=config.get(
                    "crop_size",
                    (
                        int(self.crop_size.get())
                        if len(self.crop_size.get()) > 0
                        else None
                    )
                    or 224,
                ),
                head_pose=config.get("head_pose", int(self.head_pose_checkbox.get())),
                save_folder=None,
                show_fps=config.get("show_fps", int(self.show_fps_checkbox.get())),
                use_gpu=config.get("use_gpu", int(self.use_gpu_checkbox.get())),
                use_video=False,
                use_image=False,
                limit=None,
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            error_label.configure(text=e)


@click.command()
def main():
    """Run the dot UI."""

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
