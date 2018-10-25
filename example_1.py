#!/usr/bin/env python3
# reverb simulation example
import reverbsimulator as rs
import time

# set to true to animate the bounces
do_anim = False

# create the scene
scene = rs.Scene()

# this is the main wall, a square
scene.add_obj(rs.Material(
    rs.Polygon((3.0, 3.0), (-3.0, 3.0), (-3.0, -3.0), (3.0, -3.0)),
    rs.ImpulseResponse.RC_lowpass(14500),
    tag="MainWall"
))

# this is the middle pillar (which is a circle)
scene.add_obj(rs.Material(
    rs.Circle((0.0, 0.0), 2.0),
    rs.ImpulseResponse.RC_highpass(400),
    #rs.ImpulseResponse.response_filter(lambda hz: hz / 22050.0, num_pts=100),
    tag="CenterPillar"
))

# a speaker in the bottom left
scene.add_obj(rs.Speaker(
    rs.Circle((-3.0, -3.0), 0.5),
    tag="MainSpeaker"
))

# the main microphone position
scene.add_obj(rs.Mic(
    (2.5, 2.5),
    tag="MainMic"
))


anim = None
if do_anim:
    anim = rs.viz.Animator()
    # make it a bit smaller, TODO: add scene bounding rect
    anim.scale *= 0.75

    anim.draw_scene(scene)


st = time.time()
result_IR = rs.normalize(scene.create_IR(mic_tag="MainMic", Nsamples=250, max_bounces=150, anim=anim).flattened())
print ("gen took %f s" % (time.time() - st))

st = time.time()
rs.write_wav("impulse_example_1.wav", result_IR)
print ("output took %f s" % (time.time() - st))

# uncomment this line to graph the impulse response
rs.viz.graph_samples(result_IR)

# uncomment this line to see the frequency response
rs.viz.graph_FFT(result_IR)

# you can also use this on just IRs without running a simulation
#rs.graph_FFT(rs.ImpulseResponse.response_filter(lambda hz: 1.0 * hz / 22050.0, num_pts=512))
#rs.graph_FFT(rs.ImpulseResponse.RC_highpass(12000.0, 1000))

