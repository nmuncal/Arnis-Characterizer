# Arnis-Characterizer

## Requirements

To run the program, you'll need to install the following dependencies:

- `opencv-python`: For video processing and handling image frames.
- `ultralytics`: For YOLOv8 pose estimation.
- `torch`: For PyTorch-based deep learning models.
- `tensorflow`: For loading the Keras model for action recognition.
- `numpy`: For numerical operations.
- `pickle`: For saving and loading models or data.
- `argparse`: For handling command-line arguments.
- `datetime`: For logging timestamps.
- `csv`: For storing and processing action sequences.
- `collections`: For handling frequency counts.
- `torch.nn`: For defining and using neural network layers in PyTorch.
- `ultralytics.utils.plotting.Annotator`: For visualizing results.

To install all dependencies, run:

```bash
pip install opencv-python ultralytics torch tensorflow numpy pickle argparse 
```

## Running the Program

You can run the program via the command line. Below is the basic structure for how to use it.

```bash
python main.py --source [webcam|video] --action [extract|action_recognition] --video_path [video_file_path]
```

