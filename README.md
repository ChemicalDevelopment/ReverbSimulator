# ReverbSimulator

A tool to simulate reverb given 2D geometry


## Build/Install

TODO: Make this available via `pip3 install reverbsimulator`

For now, clone this, run `pip3 install -r requirements.txt`



## Example

Check out the file `example_1.py`, and you can run from within this folder by using `python3 ./example_1.py`

If you edit it, check out the comments describing different functions. It has the ability to visualize the sound bouncing around, graph the FFT of a sound, and graph the computed impulse.

Check out what it can generate as far as resources go:

![example 1 scene](/images/example_1_scene_diagram.png?raw=true "example 1 scene")

![example 1 anim](/images/example_1_anim.png?raw=true "example 1 animation")

![example 1 impulse](/images/example_1_impulse.png?raw=true "example 1 impulse")



## FAQs/Errors

Q: I'm getting the error:

```
error while importing viz: 'No module named 'tkinter''
Traceback (most recent call last):
  File "./example_1.py", line 42, in <module>
    anim = rs.viz.Animator()
AttributeError: module 'reverbsimulator' has no attribute 'viz'
```

A: You need tkinter support for drawing. Either disable the use of animation, or install it.


For Ubuntu, run `sudo apt-get install python3-tk`
