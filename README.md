# Slideshow Control

This repository implements a live controller for a slideshow with hand gestures. This project was part of a course on machine learning from the University of Würzburg. For a demo of what the controller can do see: https://www.youtube.com/watch?v=Cd-WS7GwtyE

## Project Structure

The project structure is as follows:
- `data/:` All the training and validation data from myself is saved here.
- `evaluations/:` Notebooks and files for hyperparameter evaluation.  The different hyperparameter choices were made using a greedy approach. First, all hyperparameters for the training data were determined using `train_evaluation.py`, then all network hyperparameters were chosen using a hyperparameter search in `network_evaluation.py` and then the best postprocessor window using `postprocessor_evaluation.py`. For more information on how we made those choices see `visualizations.ipynb`.
- `presentation/:` Pictures, videos and other resources for the presentation.
- `saved_models/:` The working version of our mandatory and optional model and the corresponding pca-instance. The saved versions of the model were trained on additional private training data from other colleagues. 
- `slideshow/:` The slideshow server using reveal.js. See section [Slideshow Server](#The-Slideshow-Server) for more information.
- `src/:` All source code
  - `constants.py:` Useful constants like paths and class names
  - `data_processor:` Processor for reading in video frames or csv-files and converting it into our preprocessing format.
  - `performance_calculator.py:` The calculator for the score that determines part of our grade. It is mostly copied from the "getting started"-repository
  - `postprocessor.py:` The postprocessor that is used to convert from predictions to events. Uses a sliding window approach.
  - `preprocessor.py:` The preprocessor we use. It applies pca and computes the differences between a frame and the frame before.
  - `utils.py:` Some useful functions that are needed in different parts of the project to avoid code duplication.
- `main.py:` The script to start our slideshow controller or test the performance of our models. The usage is documented in [the next section](#how-to-use-the-slideshow-controller)
- `train.ipynb` and `train_optional.ipynb:` The notebooks used to train the current version of our models. Since not all training data is available in this repository, the results will probably differ.

## How to use the slideshow controller

First, make sure all requirements are installed. Either use `make install-deps` or `pip install -r requirements.txt`. After that, the slideshow-controller is ready to use.  
The slideshow controller supports two modes: `test` and `live`  

### Live Mode

To start the live mode the following command can be used:
```shell
python main.py --live --GESTURES_TO_USE
```
This will automatically start the reveal.js server on [localhost:8000](http://localhost:8000) and the webcam with a preview window, if a camera is available.  
With `GESTURES_TO_USE` you can specify the model that should be used. `--mand` uses the model trained on the mandatory requirements which support *swipe_left*, *swipe_right* and *rotate*. With `--opt` you can additionally use *rotate_left*, *flip_table*, *pinch*, *spread* and *point*.

### Test Mode

The test mode can be used with:
```shell
python main.py --test --GESTURES_TO_USE --src SRC_PATH --dest DEST_PATH
```
This mode reads a .csv-file from `SRC_PATH`, predicts the events and saves the .csv-file with an additional column `events` to `DEST_PATH`.  
`GESTURES_TO_USE` can also be used to specify which gestures should be used - only the mandatory or also the optional gestures - as in live mode.

## The slideshow server

The server uses Sanic with the Reveal.js slideshow which can be controlled with gestures. The server will run locally on http://0.0.0.0:8000/ or http://localhost:8000/ .

### Prerequisites

The server has been tested on Python 3.9 and needs the requirements listed in `requirements.txt` which can be installed via pip with:
```
pip install -r requirements.txt
```

### How to run?

The server can be started/stopped by starting/stopping the `slideshow_server.py` script.

### How to send events?

As soon as you go to http://localhost:8000/ in your webbrowser, a websocket connection is opened.

The server listens to GET-requests sent to http://localhost:8000/send_event, e.g. http://localhost:8000/send_event?event=right will result in a slide to the right.

The event will be forwarded to each open websocket connection, i.e. if there are multiple browser windows opened, the reveal slideshow will slide to the right in each of these windows.

### List of possible events

http://localhost:8000/send_event?event={event_name}

| event_name   | Effect                                                                  | Gesture     |
|--------------|-------------------------------------------------------------------------|-------------|
| right        | Transition to the next slide                                            | swipe_left  |
| left         | Transition to the previous slide                                        | swipe_right |
| rotate       | Rotate rotatables (-> especially images) on current slide 90° clockwise | rotate      |
| rotate_left  | Same as rotate, but counter-clockwise                                   | rotate_left |
| up           | Switch to the slide above (if any)                                      | swipe_down  |
| down         | Switch to the slide below (if any)                                      | swipe_up    |
| point        | Start/Stop video on the current slide (if any)                          | point       |
| flip_table   | Rotate rotatables on current slide 180°                                 | flip_table  |
| zoom_in (*)  | Increase zoom by 20%                                                    | spread      |
| zoom_out (*) | Decrease zoom by 20%                                                    | pinch       |

*Not working in Mozilla Firefox, but successfully tested on Safari 15.3 and Chrome 99.0.4844.83
