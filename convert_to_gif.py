from moviepy import *

clip = VideoFileClip("simulation_recording.mp4")
clip.write_gif("sim.gif")