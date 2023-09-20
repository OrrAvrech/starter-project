# starter-project

## Setup
**(Recommended - conda + pip-tools):**
1. Install a new conda environment:
```commandline
$ conda env create -f env/env.yml
```
2. Activate environment:
```commandline
$ conda activate starter-project
```
3. Sync already compiled requirements:
```commandline
$ pip-sync env/requirements.txt
```
Working environment is now ready to use. The recommended way to add new packages, is to edit `env/requirements.in` and run:
```commandline
$ pip-compile env/requirements.in
```
This line will generate an updated version of the project's `requirements.txt` file, which can be easily synced to the virtual environment with `pip-sync`.

## Data
The data collection script scrapes videos from YT using [yt-dlp](https://github.com/yt-dlp/yt-dlp), and for each video it extracts the audio and transcribes the speech.
`cd data` and execute the following:
```commandline
$ python data_collection.py --config_path data_collection.yaml
```

`data_collection.yaml` config example:
```yaml
scraper: # yt-dlp API
  extractor: "ytsearch"
  prefix_prompt: "common mistakes in"
  restrict_filenames: true
  min_vid_duration: 15
  max_vid_duration: 900
  ext: "mp4"
  no_playlist: true
  desired_agg_duration: 450
  quiet_mode: false

transcriber: # ASR pipeline 
  chunk_length_s: 10

actions:
  - push ups
  - squats
  - yoga
  - golf swing
  - ballet
  - swimming
  - slacklining
  - hitting baseball
  - strumming guitar
  - sidekick
```

## Zero-shot classification
Given a list of exercises to classify and a list of "negative actions" (found in `action_cls/other_exercises.txt`),
this scripts iterates over all videos in a given directory, runs zero-shot classification on each video, and saves the results.
`cd action_cls` and run the following:
```commandline
$ python clip_zero_shot.py --config_path zero_shot.yaml
```

`zero_shot.yaml` config example:
```yaml
vid_dir: <VIDEO_DIR_TO_EVALUATE>
data_config_path: "$PROJECT_DIR/data/data_collection.yaml"
num_frames: 8 # for model input
sample_rate: 5 # sample one every X frames
random_sample: true # randomly sample a start-time per video
window_size: 15 # sample a window per video [in seconds]
output_dir: <RESULTS_DIR_TO_SAVE_VIDEOS>
```